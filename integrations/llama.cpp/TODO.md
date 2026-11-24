> Role: tasks. Current as of 2026-11-24. History: `git log -- integrations/llama.cpp/TODO.md`.

# `ggml-spiral` tasks

Open work only. Design and matchers are in [`DESIGN.md`](DESIGN.md), build
steps in [`README.md`](README.md), numbers in
[`../../docs/measurements.md`](../../docs/measurements.md).

## Repository hygiene

- [x] Backend sources and this task list live in `integrations/llama.cpp`, not
      in the llama.cpp tree.
- [x] `SPIRAL_PROGRAM_DIR` defaults to the sibling `src/program`.
- [x] `llama-hooks.patch` is the only tracked llama.cpp delta.
- [ ] Re-verify `llama-hooks.patch` on each llama.cpp rebase: the registry
      include/registration and two CMake lines are the whole seam.
- [ ] Reconfigure `build-spiral`/`build-both` after a move or rebase; their
      CMake caches hold absolute paths.
- [ ] Decide whether to upstream the backend once the NPU path is competitive;
      until then it stays a patch on a local checkout.

## Verification still open

- [ ] Verify M=1, M=17 and K=6144/12288 through the backend path.
- [ ] Verify the decode-graph split count and overlap behaviour.
- [ ] Test `gemma-4-E2B-it-bf16.gguf` with greedy decoding against CPU.
- [ ] Confirm no late-token divergence from bf16 dequantization; record max
      relative error and token-match lengths.
- [ ] Record resident weight bytes, per-op NPU time, pack/dispatch/sync and
      decode-graph splits per token.

## Next work

1. **Chain hardening.**
   - [ ] Pipeline the chain row chunks; decode has one chunk and the chain runs
         synchronously because the output staging is shared.
   - [ ] Share the per-op weight streams with the chain BO instead of
         duplicating ~32 MB per 12288 FFN block.
2. **Graph-level fusion.**
   - [x] Softcap epilogue.
   - [x] FFN gate+up chained with GEGLU into the down matmul.
   - [ ] Q/K/V fusion for the 15 KV layers, only if the QK-norm/RoPE ordering
         can be respected (see the aliasing rule in `DESIGN.md`).
   - [ ] Keep the per-shape fallback for shared-KV layers (Q only).
3. **Broader decode path.**
   - [ ] Measure backend splits/token and the CPU segments between them.
   - [ ] Decide whether to support more ggml ops on Spiral (RMS_NORM, GEGLU,
         ROPE) or rely on the scheduler. Device-side links belong to
         `src/program/TODO.md`; the backend only needs to stop splitting there.
4. **CPU fallback.** Softcap and GEGLU are on the device now. The remaining
   fallback is norms, RoPE, attention and small elementwise ops; do not
   optimize it before the chain/norm/attention links.

## Open questions

1. Should per-handle BO ownership stay in `PreparedWeights` (current) or move to
   a backend-side map?
2. Is the K-multiple-of-256 contract sufficient for the E2B target, or is
   generic K padding required?
3. How should combined gate+up/QKV be exposed to the ggml scheduler: a graph
   rewrite, a custom op, or a backend graph-plan callback?
4. How do we bound "productive but quiet"? `Tctl`, iGPU temperature, fan RPM and
   CPU utilisation have a repeatable harness; the acceptance bar is proposed,
   not agreed (see `docs/measurements.md`).

## Risks

- **Weight BO residency:** one resident BO per prepared handle is in place;
  combined chains duplicate their segment weights.
- **Padding:** canonical padded N=256 copies must be byte-identical to the
  reference golden cases.
- **Views/aliases:** weights must be dense and contiguous; view handling is
  deferred.
- **Scheduler isolation:** supporting only `MUL_MAT` creates many backend
  splits per token. The internal CPU backend reduces this to one split for the
  prompt graph; the decode graph still has boundaries.
- **XRT resources:** hundreds of resident BOs and one exclusive hw context must
  be validated for the native-quant weight set.
- **Numerics:** native Q4_0/Q6_K should be closer to CPU than dequantized bf16
  weights, but validate with greedy generation.
- **Head chunking:** 11 launches per token at 48 bands per chunk; a one-launch
  expansion needs a much larger weight BO and more fill parts.
- **Stale artifacts:** the old `hacking-spiral-1` design and its `fw/` output are
  not part of this backend. Do not resurrect or link them.

## Definition of done

- `ggml-spiral` builds cleanly with `-DGGML_SPIRAL=ON`.
- llama.cpp offloads all Gemma 4 E2B matmuls with
  `--device Spiral0 -ngl 999`.
- Q4_0 and Q6_K weights stay in native canonical form on the host; resident
  prepared BOs are reused across token steps.
- Greedy token output matches the CPU for a representative completion.
- pp/tg and backend splits are recorded and compared with the old prototype.
- A 300-token generation runs at a usable rate with both fan tachometers and
  `Tctl` staying near the idle baseline, recorded for CPU and NPU runs.
- The combined gate+up/QKV path has either a measured graph-level integration
  or a documented reason it remains benchmark-only.
