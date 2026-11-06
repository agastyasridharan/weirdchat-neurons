"""GPU job 1, part 2 (attribution env): transformers 5.14.1 really uses flash-linear-attention (fla) for the
Gated DeltaNet "chunk" rule, and the plain torch path for the short convolution, which matches the warning in
the original attribution logs. Then fla's kernel runs forward and backward on this GPU, and its output is compared with
transformers' own torch reference on random inputs shaped like the model's (B, T, heads, 128)."""
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M  # on a GPU node, this import loads fla if installed

chunk = M.chunk_gated_delta_rule
print("chunk rule from:", getattr(chunk, "__module__", None), "| causal_conv1d_fn:", M.causal_conv1d_fn,
      "| gated norm from:", getattr(M.FusedRMSNormGated, "__module__", None), "| fast path:", M.is_fast_path_available)
assert chunk is not None and chunk.__module__.startswith("fla"), "transformers is NOT using fla for the chunk rule"
assert M.causal_conv1d_fn is None and not M.is_fast_path_available, "expected the torch convolution, as in the original logs"
print("transformers uses fla for the delta rule and torch for the convolution, as in the original runs: OK")

torch.manual_seed(0)
B, T, H, D = 1, 512, 8, 128
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = -0.1 * torch.nn.functional.softplus(torch.randn(B, T, H, device="cuda"))   # float32 and negative, like the layer's g
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
kw = dict(g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)  # as the layer calls it
o_fla, _ = chunk(q, k, v, **kw)
o_ref, _ = M.torch_chunk_gated_delta_rule(q, k, v.detach(), **kw)
err = ((o_fla.float() - o_ref.float()).norm() / o_ref.float().norm()).item()
print(f"fla vs torch reference, forward: relative error {err:.1e}")
assert err < 5e-2, "fla and the torch reference disagree"
o_fla.float().pow(2).mean().backward()
assert v.grad is not None and torch.isfinite(v.grad).all(), "fla backward gave missing or non-finite gradients"
print("fla backward: gradients finite")
print("FLA CHECK PASS")
