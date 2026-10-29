"""vLLM worker extension: zero or scale chosen MLP neurons (columns of down_proj) and attention heads (column blocks
of o_proj / out_proj) in place, restorably.

Served with --worker-extension-cls ablate_ext.AblateExt and VLLM_SERVER_DEV_MODE=1, then
  POST /collective_rpc {"method": "set_ablation", "args": [json.dumps({layer: [neuron, ...]})]}
restores the previous ablation and applies the new one ("{}" = unablated). Zeroing column i of down_proj is the
same as zeroing neuron i's activation silu(gate_i) * up_i at every position. {"scale": a, "spec": {...}} multiplies
the columns by a instead (a > 1 amplifies whatever the neuron writes, at every position). {"heads": {layer: [h, ...]}}
(with or without "spec") does the same for attention heads: head h of a full-attention layer is columns
256h..256h+255 of self_attn.o_proj, of a Gated DeltaNet layer columns 128h..128h+127 of linear_attn.out_proj.
"""
import json
import re

import torch

PAT = re.compile(r"layers\.(\d+)\.mlp\.down_proj$")
HPAT = re.compile(r"layers\.(\d+)\.(self_attn\.o_proj|linear_attn\.out_proj)$")


class AblateExt:
    def _downs(self):
        if not hasattr(self, "_wcn_downs"):
            mods = list(self.model_runner.model.named_modules())
            self._wcn_downs = {int(PAT.search(n).group(1)): m for n, m in mods if PAT.search(n)}
            self._wcn_outs = {int(HPAT.search(n).group(1)): (m, 256 if "o_proj" in n else 128) for n, m in mods if HPAT.search(n)}
            self._wcn_saved = []
        return self._wcn_downs

    def set_ablation(self, spec: str):
        downs = self._downs()
        for w, idx, cols in self._wcn_saved:
            w[:, idx] = cols
        self._wcn_saved = []
        spec = json.loads(spec)
        wrapped = "spec" in spec or "heads" in spec
        scale = float(spec.get("scale", 0.0)) if wrapped else 0.0
        neurons, heads = (spec.get("spec", {}), spec.get("heads", {})) if wrapped else (spec, {})
        cols = [(downs[int(l)].weight.data, sorted(set(idx))) for l, idx in neurons.items()]
        for l, hs in heads.items():
            m, hd = self._wcn_outs[int(l)]
            cols.append((m.weight.data, [h * hd + j for h in sorted(set(hs)) for j in range(hd)]))
        for w, idx in cols:
            idx = torch.tensor(idx, device=w.device)
            self._wcn_saved.append((w, idx, w[:, idx].clone()))
            w[:, idx] = self._wcn_saved[-1][2] * scale
        torch.cuda.synchronize()
        return {"layers": len(downs), "attn_layers": len(self._wcn_outs), "shape": list(downs[0].weight.shape),
                "zeroed": sum(len(set(v)) for v in neurons.values()), "heads": sum(len(set(v)) for v in heads.values()),
                "scale": scale}
