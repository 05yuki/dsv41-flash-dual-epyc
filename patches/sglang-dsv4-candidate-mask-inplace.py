"""The two-level indexer's prefill candidate masks are written into one table
per request instead of concatenated from the slices.

_low_ratio_index_topk_dense builds the level-one mask of a prefill chunk slice
by slice (SGLANG_DSV41_INDEXER_QUERY_STEP rows at a time) and then joined them
with torch.cat, so for a moment the pieces and the joined copy were both alive:
two [rows, compressed prefix] bool tables. At chunk 6144 that is 0.4 GB each
for a 261K-token prompt, and a ~512K-token prompt ran out of memory on the
1.39 GiB the join asked for (10-06, V4.1 under the prefill lend). Each slice's
mask now goes straight into its rows of a table allocated once; the values
are the same.

Usage: python sglang-dsv4-candidate-mask-inplace.py [sglang tree]   (default source/sglang-dsv41)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/sglang-dsv41"
F = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
s = F.read_text()
MARK = "candidate mask of this request, written slice by slice"
if MARK in s:
    print(f"already patched {F.name}")
    raise SystemExit(0)


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:300]))
    return text.replace(old, new)


s = once('''                j = torch.arange(lc, device=device)
                pieces = []
                for s0 in range(0, t_len, step):
''', '''                j = torch.arange(lc, device=device)
                # the candidate mask of this request, written slice by slice
                # (no torch.cat of the pieces: that held two copies at once)
                mask_b = (
                    torch.empty((t_len, lc), dtype=torch.bool, device=device)
                    if publish is not None and two_level
                    else None
                )
                for s0 in range(0, t_len, step):
''', s)
s = once('''                            pieces.append(
                                select_candidate_blocks(
                                    scores,
                                    lens2,
                                    topk_blocks=indexer.candidate_topk_blocks,
                                    block_size=indexer.candidate_block_size,
                                )
                            )
''', '''                            mask_b[s0 : s0 + scores.shape[0]] = select_candidate_blocks(
                                scores,
                                lens2,
                                topk_blocks=indexer.candidate_topk_blocks,
                                block_size=indexer.candidate_block_size,
                            )
''', s)
s = once('''                if publish is not None:
                    publish.append(pieces[0] if len(pieces) == 1 else torch.cat(pieces))
''', '''                if publish is not None:
                    publish.append(mask_b)
''', s)
F.write_text(s)
print("patched", F.name)
