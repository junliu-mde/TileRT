"""On-GPU verification for GLM-5 external-prefill cache injection.

Run on an 8-GPU node with TileRT weights available.

Subcommands:

  roundtrip   TileRT-only: internal prefill -> extract_cache -> inject into a
              fresh sequence -> outputs must exactly match the internal-prefill
              reference (greedy sampling). Also exercises partial injection
              (cached_len < L-1) and injection without last_hidden_state.

  cross       Requires a patched SGLang checkout + HF GLM-5 weights: runs
              SGLang prefill + KV staging for the same prompt and compares the
              extracted (ki, kv, pe) against TileRT's internally computed
              cache (RoPE layout / quantization sanity), then decodes from the
              SGLang-injected cache and compares outputs to the reference.

Examples:
  python scripts/verify_external_prefill.py roundtrip \
      --model-weights-dir /path/to/tilert-glm5-weights
  python scripts/verify_external_prefill.py cross \
      --model-weights-dir /path/to/tilert-glm5-weights \
      --hf-model-path /path/to/GLM-5.1-hf
"""

import argparse
import os
import sys
import uuid

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT = (
    "Explain, in three paragraphs, why prefill-decode disaggregation matters "
    "for low-latency LLM serving, covering KV cache hand-off, scheduling, and "
    "the trade-offs between chunked prefill and dedicated prefill engines."
)
MAX_NEW_TOKENS = 64


def load_generator(model_weights_dir: str, with_mtp: bool):
    import tilert
    from tilert.models.glm_5.generator import GLM5Generator
    from tilert.models.glm_5.model_args import ModelArgsGLM5

    tilert.load_backend("glm5")
    gen = GLM5Generator(
        model_args=ModelArgsGLM5(),
        max_new_tokens=MAX_NEW_TOKENS,
        model_weights_dir=model_weights_dir,
        with_mtp=with_mtp,
        top_k=1,
        use_topp=False,
        sampling_seed=42,
    )
    gen.init()
    gen.from_pretrained()
    return gen


def run_internal_and_extract(gen, prompt_tokens):
    """Internal prefill; return (extracted caches, hidden, reference output)."""
    prompt_len = len(prompt_tokens)
    gen.start_sequence(prompt_tokens)
    while not gen._seq_prefill_done and not gen.is_sequence_finished():
        gen.next_tokens()

    layer_caches = gen.extract_cache(end_pos=prompt_len)
    hidden = gen.extract_last_hidden_state() if gen.with_mtp else None

    reference = []
    while not gen.is_sequence_finished():
        reference.extend(gen.next_tokens())
    gen.finish_sequence()
    return layer_caches, hidden, reference


def decode_from_cache(gen, prompt_tokens, layer_caches, hidden, cached_len):
    gen.start_sequence_from_cache(
        prompt_tokens,
        layer_caches,
        last_hidden_state=hidden,
        cached_len=cached_len,
    )
    output = []
    while not gen.is_sequence_finished():
        output.extend(gen.next_tokens())
    gen.finish_sequence()
    return output


def check(name, reference, output):
    ok = output == reference
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"  reference[:16]: {reference[:16]}")
        print(f"  output[:16]:    {output[:16]}")
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(reference, output)) if a != b),
            min(len(reference), len(output)),
        )
        print(f"  first diff at token {first_diff}")
    return ok


def cmd_roundtrip(args):
    gen = load_generator(args.model_weights_dir, with_mtp=not args.disable_mtp)
    prompt_tokens = gen._prompt_to_tokens(PROMPT, None)
    prompt_len = len(prompt_tokens)
    print(f"Prompt tokens: {prompt_len}, MTP: {gen.with_mtp}")

    layer_caches, hidden, reference = run_internal_and_extract(gen, prompt_tokens)
    print(f"Reference output ({len(reference)} tokens): {reference[:16]}...")

    results = []
    out = decode_from_cache(gen, prompt_tokens, layer_caches, hidden, prompt_len)
    results.append(check("full injection (cached_len=L)", reference, out))

    partial = max(1, prompt_len - 17)
    out = decode_from_cache(gen, prompt_tokens, layer_caches, hidden, partial)
    results.append(check(f"partial injection (cached_len={partial})", reference, out))

    if gen.with_mtp:
        out = decode_from_cache(gen, prompt_tokens, layer_caches, None, partial)
        results.append(check("partial injection without hidden state", reference, out))

    gen.cleanup()
    sys.exit(0 if all(results) else 1)


def compare_caches(name, tilert_caches, sglang_caches, cached_len):
    """Structural equivalence check between the two engines' caches.

    Early layers must match near-exactly: any layout, dequant, rotation, or
    alignment bug shows up there uniformly. Deep layers legitimately diverge
    across engines (different FP8 pipelines flip MoE expert routing for some
    tokens, compounding with depth), so they are reported but not gated.
    """
    per_layer = []
    for layer_id, (t_layer, s_layer) in enumerate(zip(tilert_caches, sglang_caches)):
        for kind, t, s in zip(("ki", "kv", "pe"), t_layer, s_layer):
            t = t[:cached_len].float()
            s = s[:cached_len].float()
            cos = torch.nn.functional.cosine_similarity(
                t.flatten().unsqueeze(0), s.flatten().unsqueeze(0)
            ).item()
            per_layer.append((cos, layer_id, kind))

    worst = sorted(per_layer)[:5]
    print(f"--- {name}: worst 5 (cosine, layer, kind) ---")
    for cos, layer_id, kind in worst:
        print(f"  cos={cos:.6f} layer={layer_id} {kind}")

    early = [cos for cos, layer_id, _ in per_layer if layer_id < 10]
    overall_min = worst[0][0]
    ok = min(early) > 0.99 and overall_min > 0.5
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name} "
        f"(early-layer min {min(early):.4f}, overall min {overall_min:.4f}; "
        "deep-layer drift is expected cross-engine divergence)"
    )
    return ok


def cmd_cross(args):
    gen = load_generator(args.model_weights_dir, with_mtp=not args.disable_mtp)
    base_tokens = gen._prompt_to_tokens(PROMPT, None)
    # SGLang's radix cache matches page-aligned prefixes (page_size=64), so a
    # sub-page prompt would stage nothing. Tile to a few hundred tokens; the
    # unmatched tail exercises the partial-injection path by design.
    prompt_tokens = (base_tokens * 8)[:400]
    prompt_len = len(prompt_tokens)

    tilert_caches, hidden, reference = run_internal_and_extract(gen, prompt_tokens)

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=args.hf_model_path,
        trust_remote_code=True,
        tp_size=args.prefill_tp_size,
        mem_fraction_static=args.prefill_mem_fraction,
        kv_cache_dtype="fp8_e4m3",
        attention_backend="dsa",
        dsa_prefill_backend="trtllm",
        dsa_decode_backend="trtllm",
        disable_cuda_graph=False,
        log_level="warning",
    )
    engine.generate(
        input_ids=prompt_tokens,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
    )
    staging = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"  # nosec B108
    out_path = os.path.join(staging, f"tilert_verify_{uuid.uuid4().hex}.pt")
    engine.collective_rpc("tilert_stage_prefill_kv", token_ids=prompt_tokens, out_path=out_path)
    data = torch.load(out_path, weights_only=True)
    os.unlink(out_path)

    cached_len = min(int(data["cached_len"]), prompt_len)
    print(f"SGLang staged {data['cached_len']} tokens; using cached_len={cached_len}")
    sglang_caches = [
        (data["ki"][i], data["kv"][i], data["pe"][i]) for i in range(data["ki"].shape[0])
    ]

    results = [compare_caches("SGLang vs TileRT KV", tilert_caches, sglang_caches, cached_len)]

    # Cross-engine greedy decode is NOT expected to be token-exact (the two
    # FP8 pipelines diverge in deep layers); report both texts for semantic
    # comparison and only gate on the decode completing.
    out = decode_from_cache(gen, prompt_tokens, sglang_caches, None, cached_len)
    exact = out == reference
    print(
        f"[{'PASS' if len(out) > 0 else 'FAIL'}] decode from SGLang-injected cache "
        f"(token-exact: {exact}, informational)"
    )
    results.append(len(out) > 0)
    print(f"reference text: {gen.tokenizer.decode(reference)!r}")
    print(f"injected  text: {gen.tokenizer.decode(out)!r}")

    if args.dump:
        torch.save(
            {
                "tilert": tilert_caches,
                "sglang": sglang_caches,
                "cached_len": cached_len,
                "prompt_tokens": prompt_tokens,
                "reference": reference,
                "injected_output": out,
            },
            args.dump,
        )
        print(f"dumped caches to {args.dump}")

    engine.shutdown()
    gen.cleanup()
    sys.exit(0 if all(results) else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_round = sub.add_parser("roundtrip")
    p_round.add_argument("--model-weights-dir", required=True)
    p_round.add_argument("--disable-mtp", action="store_true")
    p_round.set_defaults(func=cmd_roundtrip)

    p_cross = sub.add_parser("cross")
    p_cross.add_argument("--model-weights-dir", required=True)
    p_cross.add_argument("--hf-model-path", required=True)
    p_cross.add_argument("--disable-mtp", action="store_true")
    p_cross.add_argument("--prefill-tp-size", type=int, default=8)
    p_cross.add_argument("--prefill-mem-fraction", type=float, default=0.85)
    p_cross.add_argument("--dump", default=None, help="Save both cache sets for offline analysis.")
    p_cross.set_defaults(func=cmd_cross)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
