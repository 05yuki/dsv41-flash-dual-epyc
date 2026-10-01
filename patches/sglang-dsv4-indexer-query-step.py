"""The dense low-ratio indexer scores a prefill chunk in query slices
(SGLANG_DSV41_INDEXER_QUERY_STEP, default 512 rows; 0 = whole chunk).

_low_ratio_index_topk_dense scored all 2048 queries of a chunk against every
compressed position so far in one [2048, prefix] fp32 matrix, then ran the
candidate selection and the top-k over it. That matrix grows with the prompt
(384 MiB by the 24th chunk of a 48K prompt, 8 GB at 1M) and is what a long
V4.1 prefill ran out of VRAM on (09-24). Nothing downstream needs the rows
together: the logits kernel, the candidate masks and the ragged top-k are all
per row. So the chunk is walked per request and per slice of rows, the
selected indices land in one [T, topk] buffer, and the published candidate
masks are concatenated per request as before. Same numbers, one slice's
scores live at a time.

Apply once to source/sglang-dsv41.
"""
from pathlib import Path

F = (Path.home() / "KTransformers/source/sglang-dsv41/python/sglang/srt/layers/attention/deepseek_v4_backend.py")
s = F.read_text()


def once(old, new, text):
    if text.count(old) != 1:
        raise SystemExit("anchor found %d times:\n%s" % (text.count(old), old[:300]))
    return text.replace(old, new)


s = once('''_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30
''', '''_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30
# rows of a prefill chunk scored at a time by the dense low-ratio indexer
# (patches/sglang-dsv4-indexer-query-step.py); 0 scores the whole chunk at once
_INDEXER_QUERY_STEP = int(__import__("os").environ.get("SGLANG_DSV41_INDEXER_QUERY_STEP", "512"))
_INDEXER_QUERY_CHECK = __import__("os").environ.get("SGLANG_DSV41_INDEXER_QUERY_CHECK") == "1"
''', s)

old = '''        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        logits = _dense_fp4_mqa_logits(
            (q_fp4, q_sf),
            (k_fp4, k_sf),
            weights,
            ks,
            ks + compress_lens,
            # the fused top-k reads score rows through 16-byte vectors
            ceil_align(max(lc_per_req), 4),
        )
        if indexer.is_candidate_source or indexer.uses_candidates:
            self._publish_or_consume_candidates(
                indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
            )
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(
            logits, compress_lens, out_offsets=ks, out_indices=selected
        )
        if indexer.uses_candidates and not indexer.is_candidate_source:
            selected = _mask_topk_scores(logits, selected, ks)
'''
new = '''        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        two_level = indexer.is_candidate_source or indexer.uses_candidates
        if _INDEXER_QUERY_STEP <= 0:
            logits = _dense_fp4_mqa_logits(
                (q_fp4, q_sf),
                (k_fp4, k_sf),
                weights,
                ks,
                ks + compress_lens,
                # the fused top-k reads score rows through 16-byte vectors
                ceil_align(max(lc_per_req), 4),
            )
            if two_level:
                self._publish_or_consume_candidates(
                    indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
                )
            topk_transform_ragged_v2(
                logits, compress_lens, out_offsets=ks, out_indices=selected
            )
            if indexer.uses_candidates and not indexer.is_candidate_source:
                selected = _mask_topk_scores(logits, selected, ks)
        else:
            # per request, per slice of rows: one slice's [rows, prefix] scores
            # live at a time instead of the whole chunk's (384 MiB at 48K)
            step = _INDEXER_QUERY_STEP
            publish = [] if indexer.is_candidate_source else None
            tok_start = 0
            for b, (lc, t_len) in enumerate(zip(lc_per_req, q_lens_cpu)):
                r0 = tok_start
                tok_start += t_len
                if t_len == 0 or lc == 0:
                    if t_len:
                        selected[r0 : r0 + t_len].fill_(-1)
                    if publish is not None:
                        publish.append(empty_mask)
                    continue
                j = torch.arange(lc, device=device)
                pieces = []
                for s0 in range(0, t_len, step):
                    rows = slice(r0 + s0, r0 + min(s0 + step, t_len))
                    lens = compress_lens[rows]
                    ks_rows = ks[rows]
                    logits = _dense_fp4_mqa_logits(
                        (q_fp4[rows], q_sf[rows]),
                        (k_fp4, k_sf),
                        weights[rows],
                        ks_rows,
                        ks_rows + lens,
                        ceil_align(lc, 4),
                    )
                    if two_level:
                        scores = logits[:, :lc]
                        if publish is None:
                            scores.masked_fill_(
                                ~self.candidate_masks[b][s0 : s0 + scores.shape[0]], -torch.inf
                            )
                        else:
                            lens2 = lens[:, None]
                            scores.masked_fill_(j[None, :] >= lens2, -torch.inf)
                            pieces.append(
                                select_candidate_blocks(
                                    scores,
                                    lens2,
                                    topk_blocks=indexer.candidate_topk_blocks,
                                    block_size=indexer.candidate_block_size,
                                )
                            )
                    out = selected[rows]
                    topk_transform_ragged_v2(logits, lens, out_offsets=ks_rows, out_indices=out)
                    if indexer.uses_candidates and not indexer.is_candidate_source:
                        out.copy_(_mask_topk_scores(logits, out, ks_rows))
                if publish is not None:
                    publish.append(pieces[0] if len(pieces) == 1 else torch.cat(pieces))
            if publish is not None:
                self.candidate_masks = publish
            if _INDEXER_QUERY_CHECK:
                # the whole-chunk path on the same inputs, row by row
                ref = torch.empty_like(selected)
                logits = _dense_fp4_mqa_logits(
                    (q_fp4, q_sf), (k_fp4, k_sf), weights, ks, ks + compress_lens,
                    ceil_align(max(lc_per_req), 4),
                )
                if two_level:
                    saved = self.candidate_masks
                    self._publish_or_consume_candidates(
                        indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
                    )
                    if publish is not None:
                        same_masks = all(
                            a.shape == b_.shape and bool((a == b_).all())
                            for a, b_ in zip(saved, self.candidate_masks)
                        )
                        self.candidate_masks = saved
                    else:
                        same_masks = True
                else:
                    same_masks = True
                topk_transform_ragged_v2(logits, compress_lens, out_offsets=ks, out_indices=ref)
                if indexer.uses_candidates and not indexer.is_candidate_source:
                    ref = _mask_topk_scores(logits, ref, ks)
                a = selected.masked_fill(selected < 0, torch.iinfo(torch.int32).max).sort(dim=-1).values
                b_ = ref.masked_fill(ref < 0, torch.iinfo(torch.int32).max).sort(dim=-1).values
                bad = int((a != b_).any(dim=-1).sum())
                print(f"[indexer-check] layer {layer.layer_id} ratio {ratio} rows {num_tokens} "
                      f"lc {lc_per_req} step {step}: rows differing {bad}, masks same {same_masks}, "
                      f"source {indexer.is_candidate_source} uses {indexer.uses_candidates}", flush=True)
'''
s = once(old, new, s)
F.write_text(s)
print("patched", F.name)
