"""Check SM120 attention with poisoned padding; requires GPUs but no model weights."""

import json

import torch
from sglang.kernels.ops.attention import flash_mla_sm120 as sm120
from sglang.srt.runtime_context import get_resources


def cache(device, page_size, value):
    stride = ((page_size * 584 + 575) // 576) * 576
    raw = torch.zeros(2, stride, dtype=torch.uint8, device=device)
    # Physical page 0 is reserved; its dummy-token payload need not be valid.
    raw[0].fill_(255)
    data = raw[1, :page_size * 576].view(page_size, 576)
    data[:, :448] = torch.full(
        (page_size, 448), value, dtype=torch.float8_e4m3fn, device=device
    ).view(torch.uint8)
    raw[1, page_size * 576:page_size * 584].fill_(127)  # UE8M0 scale 1.
    return raw.view(torch.float8_e4m3fn).as_strided(
        (2, page_size, 1, 584), (stride, 584, 584, 1)
    )


def main():
    assert torch.cuda.device_count() > 0, "This test requires an SM120 GPU."
    with torch.inference_mode():
        for ordinal in range(torch.cuda.device_count()):
            device = torch.device("cuda", ordinal)
            torch.cuda.set_device(device)
            primary, extra = cache(device, 256, 1), cache(device, 128, 2)
            tokens, heads = 65, 32  # Exercise prefill, not the <=64-token decode path.
            q = torch.zeros(tokens, 1, heads, 512, dtype=torch.bfloat16, device=device)
            indices = torch.full((tokens, 1, 128), -1, dtype=torch.int32, device=device)
            extra_indices = torch.full((tokens, 1, 512), -1, dtype=torch.int32, device=device)
            indices[:, :, 0], extra_indices[:, :, 0] = 256, 128
            lengths = torch.ones(tokens, dtype=torch.int32, device=device)
            sink = torch.zeros(heads, dtype=torch.float32, device=device)
            # Allocate both persistent buffers, then poison them before every replay.
            for source, size, tag in ((primary, 256, "primary"), (extra, 128, "extra")):
                sm120._split_kv_pages_to_64(source.view(torch.uint8), size, buffer_tag=tag)
            expected = torch.zeros_like(q)
            expected[..., :448] = 1  # (primary 1 + extra 2) / (two keys + sink).

            def attention():
                return sm120.flash_mla_with_kvcache_sm120(
                    q=q, k_cache=primary, indices=indices, topk_length=lengths,
                    attn_sink=sink, head_dim_v=512, softmax_scale=512**-0.5,
                    extra_k_cache=extra, extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=lengths,
                )[0]

            def poison():
                for tag in ("primary", "extra"):
                    get_resources().buffers[f"flash_mla_sm120_split:{device}:{tag}"].fill_(255)

            def check(result, **metadata):
                nonfinite = int((~torch.isfinite(result)).sum())
                error = float((result.float() - expected.float()).abs().max())
                print(json.dumps({"device": str(device), **metadata,
                                  "nonfinite": nonfinite, "max_abs_error": error}), flush=True)
                assert nonfinite == 0
                torch.testing.assert_close(result, expected, rtol=0, atol=0.01)

            for repeat in range(3):
                poison()
                check(attention(), repeat=repeat)
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.graph(graph, stream=capture_stream):
                captured = attention()
            torch.cuda.current_stream(device).wait_stream(capture_stream)
            for repeat in range(3):
                poison()
                graph.replay()
                check(captured, graph_replay=repeat)


if __name__ == "__main__":
    main()
