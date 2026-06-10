# HANDOFF — SAE-feature lesion on a real model (remote GPU)

This document is the single entry point for the agent taking over the **SAE + remote-GPU**
portion. It assumes no memory of how the repo got here.

## TL;DR

The whole study lives on branch **`claude/lat-literature-critique-v9z3m5`** (push there).
The probe to take to the GPU is **`real/sae_lesion.py`** — a paradigm-native lesion that
treats **SAE features as concepts on a fixed pretrained LM**, ablates each feature's
decoder direction from the residual stream, and measures collateral damage to everything
else. It already runs end-to-end on CPU (distilgpt2 + a small self-trained SAE). **Your
job: point it at a real model + a pretrained SAE on GPU.** It needs only ONE model (no
training-regime comparison), so it is the cheapest useful thing to run at scale.

## Why this probe (the scientific through-line)

Across a toy autoencoder, a small transformer, and now distilgpt2, the recurring,
substrate-independent finding is: **how clean a latent edit (steering/ablation) is, is a
measurable property of the representation — dominated by how much the model RELIES on the
concept (firing frequency / magnitude), with direction-interference second-order.** This
is the in-paradigm law that matters for real LLMs, where you don't retrain — you steer/
ablate a fixed model. The GPU rung tests whether it holds for real pretrained SAE features
at scale. Full context: `RESULTS.md` §"Paradigm-native: SAE-feature lesion on a fixed
pretrained LM" and §"Realer substrate"; `CRITIQUE.md` (P1 entries).

## What the probe measures (the contract)

For each SAE feature `f` with unit decoder direction `d_f`, it projects `d_f` out of the
residual stream at `LAYER` for all tokens (a forward hook), recomputes per-token
cross-entropy, and splits the change:
- **self_effect** = mean ΔCE over tokens where `f` fired (the legitimate edit landed).
- **collateral** = mean ΔCE over tokens where `f` did NOT fire (entanglement damage).
- **reliance** = `f`'s firing frequency; **interference** = mean cos² of `d_f` with other
  alive features' decoder dirs.

Output: `results/sae_lesion.csv` (one row/feature) + `results/sae_lesion_summary.md` with
`r(collateral, reliance)`, `r(collateral, interference)`, and the collateral/self ratio.
CPU dry-run numbers to reproduce/beat: r(collateral, reliance)=+0.52 (genuine +0.61),
interference ~flat (+0.02, expected at large dict), collateral/self ratio 0.14.

## Exactly what to change for the GPU rung

All knobs are at the top of `real/sae_lesion.py`. In priority order:

1. **`MODEL_NAME`** → your real model (e.g. a Llama/Gemma/Pythia checkpoint).
2. **`build_sae(acts, d)`** → replace the on-the-fly SAE training with a **pretrained SAE
   load** (e.g. `sae_lens`). It must return an object exposing:
   - `.encode(x)` → `(N, n_dict)` feature activations,
   - `.dec_dirs()` → `(n_dict, d)` **unit** decoder directions.
   Match the SAE's hook point to `LAYER` (and `hook_resid_post` vs `pre`).
3. **`.to(device)`** — add CUDA plumbing: `model.to("cuda")`, and move `tokens`/`acts`/the
   ablation direction to the same device. The code is plain PyTorch; no other changes.
4. **⚠ Architecture-specific ablation hook (most likely break point).** `per_token_ce`
   registers the hook on **`model.transformer.h[LAYER]`** and unpacks a tuple output —
   this is **GPT-2-specific**. For Llama/Gemma it is `model.model.layers[LAYER]` and the
   block returns a tuple whose `[0]` is the hidden state; verify the output structure and
   adjust the hook. Sanity check: ablating a high-frequency feature must raise loss
   (self_effect > 0); if all self_effects are ~0 the hook isn't biting.
5. **Corpus** — `_get_corpus()` fetches tiny-shakespeare (single-domain, fine for a CPU
   dry run only). For real features use the SAE's training distribution (e.g. the Pile /
   wikitext); diverse text matters for meaningful features and a fair interference test.

Sizing knobs: `LAYER`, `DICT_MULT`/`TOPK` (ignored if you load a pretrained SAE),
`N_TRAIN_SEQ`, `N_EVAL_SEQ`, `SEQ_LEN`, `N_FEATURES_LESION` (cost = one forward per
ablated feature; raise once on GPU).

## Known gaps / caveats to address (don't inherit them silently)

- **Single-domain corpus + tiny self-trained SAE** in the dry run → features are not
  research-grade. Use a pretrained SAE on its real corpus.
- **Interference came out flat (r≈+0.02)** because a large overcomplete dictionary makes
  decoder directions near-uniformly low-overlap. Re-test interference at the dictionary
  widths used in practice; it may matter more (or less) than reliance there.
- **self/collateral split** uses "feature fired at token t" vs "predict token t+1". Fine,
  but revisit alignment if you change the readout position convention.
- **One ablated direction at a time.** Real edits sometimes ablate a subspace; extend if
  needed.

## Open questions only the GPU rung can settle

1. Does the **reliance→collateral law** hold for real pretrained SAE features at scale,
   and does **interference** become a real second predictor at realistic dict widths?
2. (Stretch, needs training not just inference) Does the **input-AT > LAT** capability-
   modification advantage — found in the toy and the small transformer — survive when the
   three regimes are *full pretraining/fine-tuning runs of a real LLM*? This is the part
   that genuinely needs the GPU for *training*, not just inference. See `real/` (the
   three-regime transformer) for the template; `real/train.py` is plain PyTorch.

## Repro / environment

```bash
git checkout claude/lat-literature-critique-v9z3m5
pip install -r requirements.txt          # adds `transformers` (only sae_lesion needs it)
python -m real.sae_lesion                # CPU dry run: distilgpt2 + small SAE
```
CPU-only here was a deliberate constraint (no GPU until the cluster is wired over SSH);
that is why the three-regime comparison used a custom-trained transformer rather than a
real LLM. Outputs land in `results/`. Run logs (`results/*_run.log`) are git-ignored.

## Map of the work (for orientation)

| area | files | results |
|---|---|---|
| toy study + critique | `run.py` `analyze.py` `train.py` `metrics.py` `followups.py` | `RESULTS.md` `CRITIQUE.md` `results/*.csv/png` |
| capability modification (toy) | `innovability.py` `lesion.py` | `results/innovability*`, `results/lesion*` |
| selectivity (is LAT uniform?) | `selectivity.py` `real/selectivity.py` | `results/selectivity*`, `results/tx_selectivity.csv` |
| realer substrate (transformer, 3 regimes) | `real/{task,model,train,probes,run,summary}.py` | `results/tx_*` |
| **SAE-feature lesion (your start point)** | **`real/sae_lesion.py`** | `results/sae_lesion.{csv,png,summary.md}` |

Findings are written to be discoverable (each `results/*_summary.md` states its own
verdict; `RESULTS.md`/`CRITIQUE.md` carry the prose). The revised blog post is NOT drafted
— that is a separate downstream task; the results are staged for whoever writes it.
