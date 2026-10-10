"""Check BF16 CUTLASS clamp semantics on each visible GPU; no model weights needed."""

import json

import torch
from flashinfer.fused_moe import cutlass_fused_moe


def check_device(device):
    tokens, hidden, intermediate = 4, 128, 128
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device=device)
    x[:, 0] = 1
    # CUTLASS stores [up; gate]. Only one intermediate/output coordinate is active.
    w13 = torch.zeros(1, 2 * intermediate, hidden, dtype=torch.bfloat16, device=device)
    w13[0, 0, 0] = 30
    w13[0, intermediate, 0] = 20
    w2 = torch.zeros(1, hidden, intermediate, dtype=torch.bfloat16, device=device)
    w2[0, 0, 0] = 1
    ids = torch.zeros(tokens, 1, dtype=torch.int32, device=device)
    scales = torch.ones(tokens, 1, dtype=torch.float32, device=device)
    results = {}
    for name, limit in (("unclamped", None), ("clamped", 10.0)):
        clamp = None if limit is None else torch.full((1,), limit, device=device)
        out = cutlass_fused_moe(
            input=x,
            token_selected_experts=ids,
            token_final_scales=scales,
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=torch.bfloat16,
            quant_scales=None,
            swiglu_limit=clamp,
            tune_max_num_tokens=4,
        )
        torch.cuda.synchronize(device)
        if isinstance(out, (list, tuple)):
            out = out[0]
        results[name] = out[:, 0].float().cpu().tolist()
    print(json.dumps({"device": device, **results}), flush=True)
    assert all(abs(value - 600) < 1 for value in results["unclamped"])
    assert all(abs(value - 100) < 1 for value in results["clamped"])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit(
            "This optional operator check requires a CUDA GPU and FlashInfer"
        )
    for index in range(torch.cuda.device_count()):
        check_device(f"cuda:{index}")
