"""Method M4 (RelP) for Qwen3.6-27B under transformers 5.14.1: backward rules that freeze the nonlinear parts.

Every rule changes only the backward pass. Forward values stay exactly the same,
because each rule either detaches a factor that multiplies the input (x * c.detach()), or adds a term that is zero in
the forward pass (value.detach() + (x - x.detach()) * slope.detach()). The attribution itself is unchanged from
the repo's code: A = a * dM/da at the down_proj input, summed over positions (code/attrib.py:82-97); only dM/da differs.

  RMSNorm (every Qwen3_5RMSNorm, and the DeltaNet gated norm): 1/rms is frozen.
  SiLU in the MLP and after the DeltaNet convolution: the sigmoid is frozen, so the slope is sigmoid(x).
  MLP product silu(gate) * up: the half rule, 0.5 * (s * up.detach() + s.detach() * up).
  Full attention: the softmax weights are frozen (query and key detached before the attention function), and so is
    the output-gate sigmoid.
  Gated DeltaNet: query, key, decay g and beta are detached before the delta-rule kernel, and the output gate
    silu(z) and 1/rms in its gated norm are frozen, so the gradient flows along the value path only.

install(model) patches the classes and the model's own modules once; the rules apply only inside `with relp.on():`,
and outside it the patched code calls the original code. COUNTS counts the patched calls made with the rules on, so
a run can check that every module was reached. FREEZE exists only so the development tests (not included here) can switch single rules
off to show that the checks notice; the GPU job requires every entry to be True.
"""
import collections
import contextlib
import hashlib

import torch
import torch.nn.functional as TF

EXPECTED_TRANSFORMERS = "5.14.1"
# sha256 of transformers/models/qwen3_5/modeling_qwen3_5.py in the transformers 5.14.1 wheel from PyPI. The patches
# below were written against this exact file.
MODELING_SHA256 = "0e2cd8dc50885b2701d26b116c585eedcdc62a24080ec34345af55b963126ded"

STATE = {"on": False}
FREEZE = dict(rmsnorm=True, mlp_silu=True, mlp_half=True, attn_qk=True, attn_gate=True, dn_qk=True, dn_g=True,
              dn_beta=True, dn_conv_silu=True, dn_norm=True, dn_gate=True)
COUNTS = collections.Counter()
NORM_ERR = []  # per gated-norm call with the rules on: ||y - x * c|| / ||y||, y from the model's own norm module
_ORIG = {}
MQ = None  # transformers.models.qwen3_5.modeling_qwen3_5, set by install()


@contextlib.contextmanager
def on():
    prev = STATE["on"]
    STATE["on"] = True
    try:
        yield
    finally:
        STATE["on"] = prev


@contextlib.contextmanager
def _off():
    prev = STATE["on"]
    STATE["on"] = False
    try:
        yield
    finally:
        STATE["on"] = prev


def silu_frozen(x, freeze=True):
    """SiLU with the same forward value as F.silu, and slope sigmoid(x) in the backward pass."""
    y = TF.silu(x)
    if not freeze:
        return y
    return y.detach() + (x - x.detach()) * torch.sigmoid(x).detach()


def _mlp_forward(self, x):  # replaces Qwen3_5MLP.forward (modeling_qwen3_5.py:735-737)
    if not STATE["on"]:
        return _ORIG["mlp"](self, x)
    COUNTS["mlp"] += 1
    gate, up = self.gate_proj(x), self.up_proj(x)
    s = silu_frozen(gate, FREEZE["mlp_silu"])
    a = 0.5 * (s * up.detach() + s.detach() * up) if FREEZE["mlp_half"] else s * up
    return self.down_proj(a)  # down_proj is still called as a module, so the pre-hook sees a


def _rms_forward(self, x):  # replaces Qwen3_5RMSNorm.forward (modeling_qwen3_5.py:746-754), same operations
    if not STATE["on"]:
        return _ORIG["rms"](self, x)
    COUNTS["rmsnorm"] += 1
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if FREEZE["rmsnorm"]:
        r = r.detach()
    return ((xf * r) * (1.0 + self.weight.float())).type_as(x)


def _attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """Replaces Qwen3_5Attention.forward (modeling_qwen3_5.py:676-721). With the rules on, it is that code line for
    line, except that query and key are detached just before the attention function, and the gate sigmoid is
    detached."""
    if not STATE["on"]:
        return _ORIG["attn"](self, hidden_states=hidden_states, position_embeddings=position_embeddings,
                             attention_mask=attention_mask, past_key_values=past_key_values, **kwargs)
    COUNTS["attn"] += 1
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = MQ.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = MQ.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                                   MQ.eager_attention_forward)
    if FREEZE["attn_qk"]:
        query_states, key_states = query_states.detach(), key_states.detach()
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    sg = torch.sigmoid(gate)
    if FREEZE["attn_gate"]:
        sg = sg.detach()
    attn_output = attn_output * sg

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _wrap_chunk(fn):
    """Wraps a DeltaNet layer's own chunk_gated_delta_rule (an instance attribute, modeling_qwen3_5.py:423; called
    at line 539 as fn(query, key, value, g=g, beta=beta, ...))."""
    def chunk(query, key, value, g=None, beta=None, **kw):
        if STATE["on"]:
            COUNTS["dn_chunk"] += 1
            if FREEZE["dn_qk"]:
                query, key = query.detach(), key.detach()
            if FREEZE["dn_g"]:
                g = g.detach()
            if FREEZE["dn_beta"]:
                beta = beta.detach()
        return fn(query, key, value, g=g, beta=beta, **kw)
    chunk.relp_inner = fn
    return chunk


def _patch_gated_norm(mod):
    """Overrides one DeltaNet gated norm's forward (called at modeling_qwen3_5.py:559 as self.norm(x, z)). With fla
    installed this is fla's FusedRMSNormGated, which computes y = x * rstd * weight * silu(z) in float32 with
    rstd = 1 / sqrt(mean(x^2) + eps) (fla/modules/fused_norm_gate.py:66-102, fla 0.5.2). The forward value is taken
    from the module itself; the backward pass sees the same formula with rstd and silu(z) frozen. NORM_ERR records
    how far the module's output is from that formula, so a run can check the formula."""
    orig = mod.forward
    eps = mod.eps if hasattr(mod, "eps") else mod.variance_epsilon  # fla's name, or transformers' torch fallback
    act = getattr(mod, "activation", "silu")
    if act not in ("silu", "swish"):
        raise RuntimeError(f"gated norm activation is {act!r}, expected silu")
    if getattr(mod, "bias", None) is not None:
        raise RuntimeError("gated norm has a bias, which the frozen formula does not include")

    def forward(x, gate, *args, **kw):
        if not STATE["on"]:
            return orig(x, gate, *args, **kw)
        if args or kw:
            raise RuntimeError("unexpected arguments to the gated norm")
        COUNTS["dn_norm"] += 1
        with torch.no_grad(), _off():
            y = orig(x, gate)
        xf = x.float()
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        if FREEZE["dn_norm"]:
            r = r.detach()
        gz = TF.silu(gate.float())
        if FREEZE["dn_gate"]:
            gz = gz.detach()
        lin = xf * r * mod.weight.float() * gz
        yf = y.float()
        NORM_ERR.append(float((yf - lin.detach()).norm() / yf.norm().clamp_min(1e-30)))
        return y + (lin - lin.detach()).to(y.dtype)

    mod.forward = forward


def _silu_conv(x, inplace=False):
    if not STATE["on"]:
        return TF.silu(x, inplace=inplace)
    if inplace:
        raise RuntimeError("in-place SiLU with the rules on")
    COUNTS["dn_conv_silu"] += 1
    return silu_frozen(x, FREEZE["dn_conv_silu"])


class _FProxy:
    """Stands in for torch.nn.functional inside modeling_qwen3_5 only, so that the DeltaNet's F.silu after its
    convolution (modeling_qwen3_5.py:501) uses the frozen-sigmoid SiLU when the rules are on. Every other name is
    torch.nn.functional's own. COUNTS["dn_conv_silu"] shows how many calls it changed."""

    def __getattr__(self, name):
        return _silu_conv if name == "silu" else getattr(TF, name)


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def install(model):
    """Patch once, after the weights are loaded. Returns the number of patched calls expected per forward pass."""
    global MQ
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if _ORIG:
        raise RuntimeError("relp.install() was already called")
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise RuntimeError(f"transformers {transformers.__version__}; the RelP patches are for {EXPECTED_TRANSFORMERS}")
    sha = file_sha256(modeling_qwen3_5.__file__)
    if sha != MODELING_SHA256:
        raise RuntimeError(f"modeling_qwen3_5.py differs from the 5.14.1 file the patches were written for: {sha}")
    MQ = modeling_qwen3_5
    layers = model.model.layers
    n_lin = n_full = 0
    for L in layers:
        if type(L.mlp).__name__ != "Qwen3_5MLP" or type(L.mlp.act_fn).__name__ not in ("SiLUActivation", "SiLU"):
            raise RuntimeError(f"unexpected MLP or activation: {type(L.mlp).__name__}, {type(L.mlp.act_fn).__name__}")
        if L.block_type == "linear_attention":
            la = L.linear_attn
            if la.causal_conv1d_fn is not None:
                raise RuntimeError("causal_conv1d is installed, so the SiLU at modeling_qwen3_5.py:501 does not run")
            n_lin += 1
        elif L.block_type == "full_attention":
            n_full += 1
        else:
            raise RuntimeError(f"unknown block type {L.block_type}")
    # A module with its own instance-level forward (an accelerate hook, for example) would keep calling the original
    # class code and bypass the class patches below. COUNTS would show it, but stopping here is clearer.
    patched = (MQ.Qwen3_5MLP, MQ.Qwen3_5RMSNorm, MQ.Qwen3_5Attention)
    hooked = [n for n, m in model.named_modules()
              if isinstance(m, patched) and ("forward" in vars(m) or hasattr(m, "_hf_hook"))]
    if hooked:
        raise RuntimeError(f"{len(hooked)} modules have their own forward (first: {hooked[0]}); the patches would be bypassed")
    _ORIG["mlp"], MQ.Qwen3_5MLP.forward = MQ.Qwen3_5MLP.forward, _mlp_forward
    _ORIG["rms"], MQ.Qwen3_5RMSNorm.forward = MQ.Qwen3_5RMSNorm.forward, _rms_forward
    _ORIG["attn"], MQ.Qwen3_5Attention.forward = MQ.Qwen3_5Attention.forward, _attn_forward
    _ORIG["F"], MQ.F = MQ.F, _FProxy()
    for L in layers:
        if L.block_type == "linear_attention":
            L.linear_attn.chunk_gated_delta_rule = _wrap_chunk(L.linear_attn.chunk_gated_delta_rule)
            _patch_gated_norm(L.linear_attn.norm)
    n_rms = sum(type(m) is MQ.Qwen3_5RMSNorm for m in model.modules())
    return dict(mlp=len(layers), rmsnorm=n_rms, attn=n_full, dn_chunk=n_lin, dn_norm=n_lin, dn_conv_silu=n_lin)
