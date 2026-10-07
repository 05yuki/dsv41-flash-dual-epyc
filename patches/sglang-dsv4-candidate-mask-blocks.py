"""The two-level indexer's prefill candidate masks are kept per block, not per
position (on top of sglang-dsv4-candidate-mask-inplace.py).

The candidate-source layer published, per request, a [rows, compressed prefix]
bool mask, and it stayed alive for the whole chunk: at chunk 6144 that is
1.6 GB for a 261K-token prompt and 3.1 GB at 512K, which with the indexer's
logits slice (512 rows x prefix x fp32, 708 MiB at ~356K) is where a long V4.1
prefill ran out of memory (10-06). select_candidate_blocks already decides per
block and only then expanded the decision to every position; the sliced
prefill path now keeps the [rows, blocks] decision (block_size times smaller)
and the consumer layers sink the non-candidate blocks through a
[rows, blocks, block_size] view of their scores, without an expanded copy.
The same positions end up -inf. Masks that come as positions (the whole-chunk
path, decode) are applied as before.

Usage: python sglang-dsv4-candidate-mask-blocks.py [sglang tree]   (default source/sglang-dsv41)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "KTransformers/source/sglang-dsv41"
srt = root / "python/sglang/srt/layers/attention"
B = srt / "deepseek_v4_backend.py"
X = srt / "dsv4/indexer.py"
MARK = "_sink_non_candidates"


def once(old, new, text, name):
    if text.count(old) != 1:
        raise SystemExit("%s: anchor found %d times:\n%s" % (name, text.count(old), old[:300]))
    return text.replace(old, new)


b = B.read_text()
if MARK in b:
    print(f"already patched {B.name}")
    raise SystemExit(0)
if "candidate mask of this request, written slice by slice" not in b:
    raise SystemExit("apply sglang-dsv4-candidate-mask-inplace.py first")

x = X.read_text()
x = once('''    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the''', '''    topk_blocks: int,
    block_size: int,
    expand: bool = True,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the''', x, X.name)
x = once('''    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
''', '''    if not expand:
        return keep  # [..., blocks]: the decision per block, not yet per position
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
''', x, X.name)

b = once('''def _mask_topk_scores(
''', '''def _sink_non_candidates(scores: torch.Tensor, mask: torch.Tensor, block_size: int) -> None:
    """scores [n, w] in place: -inf where mask is False. mask is either per
    position [n, w] or per block [n, ceil(w / block_size)]; a block mask goes
    through a [n, blocks, block_size] view of scores, so no per-position copy
    of it is made (sglang-dsv4-candidate-mask-blocks.py)."""
    w = scores.shape[-1]
    if mask.shape[-1] == w:
        scores.masked_fill_(~mask, -torch.inf)
        return
    full = w - w % block_size
    nb = full // block_size
    if full:
        scores[:, :full].unflatten(-1, (nb, block_size)).masked_fill_(
            ~mask[:, :nb, None], -torch.inf
        )
    if full < w:
        scores[:, full:].masked_fill_(~mask[:, nb : nb + 1], -torch.inf)


def _expand_candidates(mask: torch.Tensor, width: int, block_size: int) -> torch.Tensor:
    """A per-position copy of a block mask (the check path only)."""
    if mask.shape[-1] == width:
        return mask
    return mask.repeat_interleave(block_size, dim=-1)[..., :width]


def _mask_topk_scores(
''', b, B.name)
b = once('''                mask_b = (
                    torch.empty((t_len, lc), dtype=torch.bool, device=device)
                    if publish is not None and two_level
                    else None
                )
''', '''                # per block: block_size times smaller than per position
                mask_b = (
                    torch.empty(
                        (t_len, -(-lc // indexer.candidate_block_size)),
                        dtype=torch.bool,
                        device=device,
                    )
                    if publish is not None and two_level
                    else None
                )
''', b, B.name)
b = once('''                        if publish is None:
                            scores.masked_fill_(
                                ~self.candidate_masks[b][s0 : s0 + scores.shape[0]], -torch.inf
                            )
''', '''                        if publish is None:
                            _sink_non_candidates(
                                scores,
                                self.candidate_masks[b][s0 : s0 + scores.shape[0]],
                                indexer.candidate_block_size,
                            )
''', b, B.name)
b = once('''                            mask_b[s0 : s0 + scores.shape[0]] = select_candidate_blocks(
                                scores,
                                lens2,
                                topk_blocks=indexer.candidate_topk_blocks,
                                block_size=indexer.candidate_block_size,
                            )
''', '''                            mask_b[s0 : s0 + scores.shape[0]] = select_candidate_blocks(
                                scores,
                                lens2,
                                topk_blocks=indexer.candidate_topk_blocks,
                                block_size=indexer.candidate_block_size,
                                expand=False,
                            )
''', b, B.name)
b = once('''                        same_masks = all(
                            a.shape == b_.shape and bool((a == b_).all())
                            for a, b_ in zip(saved, self.candidate_masks)
                        )
''', '''                        same_masks = all(
                            bool(
                                (
                                    _expand_candidates(
                                        a, b_.shape[-1], indexer.candidate_block_size
                                    )
                                    == b_
                                ).all()
                            )
                            if a.numel() and b_.numel()
                            else a.numel() == b_.numel()
                            for a, b_ in zip(saved, self.candidate_masks)
                        )
''', b, B.name)
b = once('''            if publish is None:
                scores.masked_fill_(~self.candidate_masks[b], -torch.inf)
                continue
''', '''            if publish is None:
                _sink_non_candidates(scores, self.candidate_masks[b], indexer.candidate_block_size)
                continue
''', b, B.name)
X.write_text(x)
B.write_text(b)
print("patched", X.name, B.name)
