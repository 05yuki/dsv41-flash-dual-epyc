"""select_candidate_blocks without the F.pad copy of the dense logits.

The low-ratio indexer scores every query of a prefill chunk against every
position so far: [2048, prefix] fp32, 384 MiB by the 24th chunk of a 48K
prompt. F.pad to a block multiple then copied the whole thing to add a few
columns, and that second 384 MiB is what a 48K prefill of V4.1 died on
(09-24, 5 and 10 GPU experts, mem_fraction 0.85-0.87 alike: "Tried to
allocate 384.00 MiB" in select_candidate_blocks). Take the block maxima over
the full blocks in place and the tail block separately; same scores, one
[T, blocks] result.

Apply once to source/sglang-dsv41.
Usage: python sglang-dsv4-candidate-blocks-nopad.py [sglang tree]   (default source/sglang-dsv41)
"""
import sys
from pathlib import Path

F = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/sglang-dsv41") / "python/sglang/srt/layers/attention/dsv4/indexer.py"
s = F.read_text()
if "no padded copy of the [T, prefix] logits" in s:
    print(f"already patched {F.name}")
    raise SystemExit(0)
old = '''    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
'''
new = '''    width = logits.size(-1)
    # no padded copy of the [T, prefix] logits (384 MiB at 48K, and the OOM of
    # a long prefill): block maxima over the full blocks, the tail on its own
    full = width - width % block_size
    scores = logits[..., :full].unflatten(-1, (-1, block_size)).amax(dim=-1)
    if full < width:
        scores = torch.cat([scores, logits[..., full:].amax(dim=-1, keepdim=True)], dim=-1)
    num_blocks = scores.size(-1)
'''
if s.count(old) != 1:
    raise SystemExit("anchor found %d times" % s.count(old))
F.write_text(s.replace(old, new))
print("patched", F.name)
