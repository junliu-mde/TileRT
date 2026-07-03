"""Offline analysis of a cross_dump.pt produced by verify_external_prefill cross --dump.

Discriminates between systematic misalignment (layer shift, row shift,
missing transform) and inherent cross-engine numerical divergence
(FP8 pipelines + MoE routing flips compounding with depth).
"""

import argparse

import torch


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0).float(), b.flatten().unsqueeze(0).float()
    ).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    args = parser.parse_args()

    d = torch.load(args.dump, weights_only=True)
    tilert, sglang, n = d["tilert"], d["sglang"], d["cached_len"]
    n_layers = len(tilert)

    print("=== per-layer cosine (aligned) ===")
    curves = {k: [] for k in ("ki", "kv", "pe")}
    for i in range(n_layers):
        for k, ti, si in zip(("ki", "kv", "pe"), tilert[i], sglang[i]):
            curves[k].append(cos(ti[:n], si[:n]))
    for k, c in curves.items():
        qs = [round(v, 3) for v in (min(c), sorted(c)[len(c) // 2], max(c))]
        print(f"  {k}: min/med/max = {qs}")
        print(f"  {k} by layer (every 6th): {[round(v, 2) for v in c[::6]]}")

    print("\n=== layer-shift test on kv (systematic misalignment check) ===")
    for shift in (-1, 0, 1):
        vals = []
        for i in range(max(0, -shift), min(n_layers, n_layers - shift)):
            vals.append(cos(tilert[i][1][:n], sglang[i + shift][1][:n]))
        print(f"  shift={shift:+d}: mean={sum(vals)/len(vals):.4f}")

    print("\n=== row-shift test on kv layer 40 (position alignment check) ===")
    t40, s40 = tilert[40][1][:n].float(), sglang[40][1][:n].float()
    for shift in (-1, 0, 1):
        a = t40[max(0, -shift) : n - max(0, shift)]
        b = s40[max(0, shift) : n - max(0, -shift)]
        print(f"  shift={shift:+d}: cos={cos(a, b):.4f}")

    print("\n=== per-token row cosine distribution at worst kv layer ===")
    worst_layer = min(range(n_layers), key=lambda i: curves["kv"][i])
    t, s = tilert[worst_layer][1][:n].float(), sglang[worst_layer][1][:n].float()
    row_cos = torch.nn.functional.cosine_similarity(t, s, dim=-1)
    hist = torch.histc(row_cos, bins=10, min=-1.0, max=1.0)
    print(f"  worst kv layer = {worst_layer} (cos={curves['kv'][worst_layer]:.3f})")
    print(f"  row-cos deciles [-1..1]: {[int(v) for v in hist.tolist()]}")
    print(f"  rows below 0.9: {(row_cos < 0.9).sum().item()}/{n}")
    print(f"  rows below 0.5: {(row_cos < 0.5).sum().item()}/{n}")
    bad = (row_cos < 0.5).nonzero().flatten().tolist()
    print(f"  worst rows (first 20): {bad[:20]}")

    print("\n=== same distribution at an early kv layer (5) ===")
    t, s = tilert[5][1][:n].float(), sglang[5][1][:n].float()
    row_cos = torch.nn.functional.cosine_similarity(t, s, dim=-1)
    print(
        f"  layer-5 kv cos={curves['kv'][5]:.4f}, rows below 0.9: {(row_cos < 0.9).sum().item()}/{n}"
    )


if __name__ == "__main__":
    main()
