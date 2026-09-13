"""Turn an expert distribution dump into a static expert placement.

Reads the ``expert_distribution_recorder_*.pt`` written by
tools/record-dsv41-routing.sh (``logical_count``: [num_layers, num_logical]),
puts each layer's ``--gpu`` hottest logical experts at physical slots 0..N-1 —
the slots kt_ep_wrapper runs on the GPU — and the rest in ascending order, and
writes ``{"physical_to_logical_map": [...]}`` for ``--init-expert-location``.

Also prints, per layer and overall, the share of routed tokens the GPU slots
would catch, which is the CPU-stage work the placement removes.
"""
import argparse
import json
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="expert_distribution_recorder_*.pt")
    ap.add_argument("--layers", type=int, default=40, help="MoE layers in the model")
    ap.add_argument("--experts", type=int, default=384, help="routed experts per layer")
    ap.add_argument("--gpu", type=int, default=7, help="experts per layer on the GPU (kt-num-gpu-experts)")
    ap.add_argument("-o", "--out", default="dsv41-expert-placement.json")
    args = ap.parse_args()

    data = torch.load(args.dump, map_location="cpu", weights_only=False)
    counts = data["logical_count"]
    # "stat" mode dumps a ring of per-step counts, [steps, layers, experts];
    # fold the steps into totals
    if getattr(counts, "ndim", 0) == 3:
        counts = counts.sum(0)
    # the KT wrapper's histogram (KT_ROUTING_DUMP) is padded to [64, 512]
    counts = counts[: args.layers, : args.experts]
    if counts.dim() != 2:
        sys.exit(f"logical_count has shape {tuple(counts.shape)}, expected [layers, experts]")
    num_layers, num_experts = counts.shape
    n = args.gpu
    if n > num_experts:
        sys.exit(f"--gpu {n} exceeds {num_experts} experts")

    p2l = []
    caught_total = 0.0
    routed_total = 0.0
    print(f"layers={num_layers} experts={num_experts} gpu_slots={n}")
    print("layer  gpu_share  hottest -> coldest of the chosen")
    for layer in range(num_layers):
        row = counts[layer].to(torch.float64)
        routed = float(row.sum())
        order = torch.argsort(row, descending=True)
        hot = order[:n].tolist()
        rest = sorted(set(range(num_experts)) - set(hot))
        p2l.append(hot + rest)
        caught = float(row[hot].sum())
        caught_total += caught
        routed_total += routed
        share = caught / routed if routed else 0.0
        print(f"{layer:5d}  {share:9.3f}  {hot}")
    overall = caught_total / routed_total if routed_total else 0.0
    print(f"overall gpu share {overall:.3f}  (uniform routing would give {n / num_experts:.3f})")

    with open(args.out, "w") as f:
        json.dump({"physical_to_logical_map": p2l}, f, separators=(",", ":"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
