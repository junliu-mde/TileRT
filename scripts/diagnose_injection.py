"""Single-load diagnosis for the external-prefill injection failure.

Loads GLM-5 weights ONCE, then runs a sequence of probes that discriminate
between the candidate root causes of the roundtrip failure:

  P0  op inventory        - native setters/getters available in this wheel
  P1  cache visibility    - does the megakernel write the Python-side cache
                            tensors, and at which rows? (pointer sharing +
                            row-offset check, dev0 vs dev3 replication)
  P2  CUR_POS liveness    - does forward() consume/advance the CUR_POS temp
                            var, i.e. is it real kernel state?
  P3  reset effect        - does dsa_show_hands_reset zero the cache buffers /
                            CUR_POS?
  P4  no-reset re-decode  - same-session decode replay without native reset:
                            is (caches + CUR_POS + valid_tokens + seed) a
                            sufficient state description at all?
  P5  post-reset restore  - after a native reset, restore state WITHOUT
                            touching caches (they still hold correct rows
                            unless P3 says otherwise): is reset the killer?
  P6  inject fidelity     - zero buffers -> inject -> extract -> byte-compare
  P7  dirty-runtime inject- full injected hand-off for prompt A on a runtime
                            that just ran prompt B with NO reset in between:
                            validates the "skip native reset, overwrite
                            everything" fix candidate for real cross-request
                            use.

Usage (on the 8-GPU pod):
  python3 diagnose_injection.py --model-weights-dir /tmp/scratch-space/GLM-5.1-FP8-TileRT
  python3 diagnose_injection.py --model-weights-dir ... --disable-mtp
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPT_A = (
    "Explain, in three paragraphs, why prefill-decode disaggregation matters "
    "for low-latency LLM serving, covering KV cache hand-off, scheduling, and "
    "the trade-offs between chunked prefill and dedicated prefill engines."
)
PROMPT_B = (
    "List five classic distributed systems papers and one sentence about why "
    "each one still matters today."
)
MAX_NEW = 24


def banner(title):
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def verdict(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def load_generator(args):
    import tilert
    from tilert.models.glm_5.generator import GLM5Generator
    from tilert.models.glm_5.model_args import ModelArgsGLM5

    tilert.load_backend("glm5")
    gen = GLM5Generator(
        model_args=ModelArgsGLM5(),
        max_new_tokens=MAX_NEW,
        model_weights_dir=args.model_weights_dir,
        with_mtp=not args.disable_mtp,
        top_k=1,
        use_topp=False,
        sampling_seed=42,
    )
    gen.init()
    gen.from_pretrained()
    return gen


def get_caches(gen, dev):
    return gen.decode_layer._get_device_result(dev)[1]


def get_intermediates(gen, dev):
    return gen.decode_layer._get_device_result(dev)[0]


def cur_pos_values(gen):
    from tilert.models.glm_5._dsa_v32.temp_var_indices import Idx

    vals = []
    for dev in range(gen.decode_layer.num_devices):
        vals.append(int(get_intermediates(gen, dev)[Idx.CUR_POS].flatten()[0].item()))
    return vals


def row_norms(gen, dev, layer, rows):
    caches = get_caches(gen, dev)
    base = layer * 3
    out = {}
    for name, idx in (("ki", 0), ("kv", 1), ("pe", 2)):
        t = caches[base + idx][0, :rows, :].float()
        out[name] = t.norm(dim=-1)
    return out


def print_row_map(tag, norms, rows):
    for name, n in norms.items():
        nz = (n > 1e-6).nonzero().flatten().tolist()
        head = nz[:4]
        tail = nz[-4:] if len(nz) > 4 else []
        print(
            f"  {tag} {name}: nonzero_rows={len(nz)}/{rows} "
            f"first={head} last={tail} norm[0]={n[0]:.3f} norm[{rows - 1}]={n[rows - 1]:.3f}",
            flush=True,
        )


def pump_prefill(gen):
    while not gen._seq_prefill_done and not gen.is_sequence_finished():
        gen.next_tokens()


def decode_n(gen, n):
    out = []
    while not gen.is_sequence_finished() and len(out) < n:
        out.extend(gen.next_tokens())
    return out[:n]


def rebuild_decode_state(gen, prompt_tokens, mtp):
    """Set Python-side sequence state to 'prefill finished, decode not started'."""
    L = len(prompt_tokens)
    gen._reset_sequence_state()
    gen._seq_active_mtp = mtp
    gen.decode_layer.set_sampling_seed(gen.sampling_seed, with_mtp=mtp)
    gen._init_sequence_tensors(prompt_tokens)
    gen._seq_prefill_done = True
    gen._seq_prefill_pos = L
    if mtp:
        gen._seq_cur_pos = L - 1
    else:
        gen._seq_prev_pos = L - 1
        gen._seq_cur_pos = L
    gen._sequence_active = True


def restore_native_position(gen, prompt_tokens, mtp):
    gen.set_cur_pos(len(prompt_tokens) - 1)
    if mtp:
        gen.decode_layer.set_prefill_valid_tokens(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-weights-dir", required=True)
    parser.add_argument("--disable-mtp", action="store_true")
    args = parser.parse_args()

    gen = load_generator(args)
    mtp = gen.with_mtp
    tokens_a = gen._prompt_to_tokens(PROMPT_A, None)
    tokens_b = gen._prompt_to_tokens(PROMPT_B, None)
    L = len(tokens_a)
    print(f"\nPrompt A: {L} tokens, MTP={mtp}", flush=True)
    results = {}

    banner("P0 native op inventory")
    ops = sorted(op for op in dir(torch.ops.tilert) if not op.startswith("_"))
    interesting = [
        op
        for op in ops
        if any(k in op for k in ("set", "reset", "get", "cur", "pos", "seq", "len"))
    ]
    print(f"  {len(ops)} ops total; state-ish ops:", flush=True)
    for op in interesting:
        print(f"    {op}", flush=True)

    banner("P1 cache visibility after internal prefill")
    gen.start_sequence(tokens_a)
    pump_prefill(gen)
    rows = L + 16
    norms0 = row_norms(gen, 0, 0, rows)
    print_row_map("dev0 layer0", norms0, rows)
    print_row_map("dev0 layer77", row_norms(gen, 0, 77, rows), rows)
    print_row_map("dev3 layer0", row_norms(gen, 3, 0, rows), rows)
    kernel_writes_visible = bool((norms0["kv"][: L - 1] > 1e-6).all())
    results["P1_pointer_shared_rows_0_to_L"] = verdict(
        "P1 kernel writes visible at rows [0, L-1) on dev0",
        kernel_writes_visible,
    )
    d0 = get_caches(gen, 0)[1][0, : L - 1, :].float().cpu()
    d3 = get_caches(gen, 3)[1][0, : L - 1, :].float().cpu()
    results["P1_replicated"] = verdict(
        "P1 dev0/dev3 kv cache identical",
        torch.equal(d0, d3),
        f"max_diff={(d0 - d3).abs().max().item():.6f}",
    )

    banner("P2 CUR_POS liveness across decode")
    print(f"  CUR_POS after prefill (per device): {cur_pos_values(gen)}", flush=True)
    ext1 = gen.extract_cache(end_pos=L)
    ref = decode_n(gen, MAX_NEW)
    print(f"  reference tokens: {ref}", flush=True)
    post_decode_cur = cur_pos_values(gen)
    print(f"  CUR_POS after {len(ref)} decoded tokens: {post_decode_cur}", flush=True)
    results["P2_curpos_advances"] = verdict(
        "P2 CUR_POS advanced past L (kernel-maintained state)",
        post_decode_cur[0] > L - 1,
        f"cur_pos={post_decode_cur[0]} L={L}",
    )

    banner("P4 same-session re-decode WITHOUT native reset")
    rebuild_decode_state(gen, tokens_a, mtp)
    restore_native_position(gen, tokens_a, mtp)
    out = decode_n(gen, MAX_NEW)
    results["P4_noreset_redecode"] = verdict(
        "P4 no-reset re-decode matches reference", out == ref, f"got {out[:8]}..."
    )

    banner("P3 effect of native reset on buffers/CUR_POS")
    gen._sequence_active = True  # ensure finish_sequence performs the native reset
    gen.finish_sequence()
    ext2 = gen.extract_cache(end_pos=L)
    # Rows near L-1 are legitimately rewritten by decode forwards; only rows
    # well inside the prefix prove whether reset cleared the buffers.
    keep = L - 4
    cache_survives = all(
        torch.equal(a[:keep], b[:keep]) for t1, t2 in zip(ext1, ext2) for a, b in zip(t1, t2)
    )
    results["P3_cache_survives_reset"] = verdict(
        "P3 cache buffers survive native reset", cache_survives
    )
    print(f"  CUR_POS after reset: {cur_pos_values(gen)}", flush=True)

    banner("P5 post-reset state restore (caches untouched)")
    rebuild_decode_state(gen, tokens_a, mtp)
    restore_native_position(gen, tokens_a, mtp)
    out = decode_n(gen, MAX_NEW)
    results["P5_postreset_restore"] = verdict(
        "P5 post-reset restore matches reference", out == ref, f"got {out[:8]}..."
    )
    gen.finish_sequence()

    banner("P6 inject fidelity (zero -> inject -> extract)")
    for dev in range(gen.decode_layer.num_devices):
        caches = get_caches(gen, dev)
        for layer in range(gen.config.n_layers):
            for i in range(3):
                caches[layer * 3 + i][0, :L, :].zero_()
    gen.inject_cache([(ki[:L], kv[:L], pe[:L]) for ki, kv, pe in ext1], start_pos=0)
    ext3 = gen.extract_cache(end_pos=L)
    inject_ok = all(torch.equal(a, b) for t1, t2 in zip(ext1, ext3) for a, b in zip(t1, t2))
    results["P6_inject_fidelity"] = verdict("P6 inject writes exact bytes", inject_ok)

    banner("P7 injected hand-off on dirty runtime (no reset since prompt B)")
    # Run prompt B fully (internal path, with its own reset first via
    # start_sequence -> finish_sequence in our manual flow).
    gen.start_sequence(tokens_b)
    pump_prefill(gen)
    decode_n(gen, 8)
    # NOTE: deliberately NO finish_sequence/reset here.
    gen._sequence_active = False  # keep finish_sequence (if any) from resetting
    rebuild_decode_state(gen, tokens_a, mtp)
    gen.inject_cache([(ki[:L], kv[:L], pe[:L]) for ki, kv, pe in ext1], start_pos=0)
    restore_native_position(gen, tokens_a, mtp)
    out = decode_n(gen, MAX_NEW)
    results["P7_dirty_runtime_inject"] = verdict(
        "P7 dirty-runtime injected decode matches reference",
        out == ref,
        f"got {out[:8]}...",
    )

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(
        """
Interpretation:
  P1 FAIL                -> kernel does not write the Python cache tensors
                            (internal copies); injection API needs pointer-
                            level rework.
  P1 PASS + P4 FAIL      -> even without reset, (CUR_POS + valid_tokens) is
                            not a complete state restore; look for more state
                            in P0's op list.
  P4 PASS + P5 FAIL      -> native reset destroys kernel state that
                            set_cur_pos cannot restore; fix = defer/skip the
                            native reset for injected sequences (P7 validates
                            this works cross-request).
  P5 PASS + P6 FAIL      -> inject_cache indexing/copy bug.
  All PASS               -> state machine entry conditions in
                            start_sequence_from_cache are wrong; diff against
                            rebuild_decode_state() here.
""",
        flush=True,
    )
    gen.cleanup()


if __name__ == "__main__":
    main()
