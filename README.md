# LAT Toy-Model Study

A toy-model study testing whether latent adversarial training (LAT) gives a model
a more *robust* internal representation of a concept, and whether the
dimensionality proxies used on real models (participation ratio, SVD
concentration, SAE reconstruction) actually track ground-truth superposition.

## Documents

- [`SPEC.md`](SPEC.md) — goal, theory, model, metrics, grid, completion conditions.
- [`RESULTS.md`](RESULTS.md) — verdict on Claim B (concentrate + widen) and
  Claim C (do the proxies track ground truth?), with caveats.
- [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) — deep research mapping the
  hypothesis onto prior art.
- [`REVIEW_PROMPT.md`](REVIEW_PROMPT.md) — prompt for an agent to critically
  review the study and decide further changes.

## Run

```bash
pip install -r requirements.txt
python3 run.py             # full sweep -> results/runs.csv, results/concepts.csv
python3 analyze.py         # tables, plots, results/summary.json + verdict
python3 followups.py       # SPEC §9 forks: eps-sweep + targeted-LAT (critique)
python3 followups_summary.py  # -> results/followups_summary.md
python3 evolvability.py    # SPEC §10: Wagner robustness-enables-evolvability probe
python3 evolvability_summary.py  # -> results/evolvability_summary.md
```

See [`CRITIQUE.md`](CRITIQUE.md) for the post-literature-review revision (why the
v1 concentration verdict was a measurement artifact and how it was corrected).

## Layout

| file | role |
|---|---|
| `config.py` | all knobs (grid, seeds, ε, SAE, probe, sweep/targeted) |
| `data.py` | synthetic sparse features; single- and two-layer toy models |
| `train.py` | baseline / input-AT / LAT / targeted-LAT training (PGD) |
| `sae.py` | TopK SAE proxies (FVU, L0, monosemanticity) |
| `metrics.py` | ground-truth superposition, concept proxies, robust-region probe |
| `run.py` | sweep orchestration (grid × condition × seed) |
| `analyze.py` | summary tables, plots, B/C verdict (+ weight-level concentration) |
| `followups.py` | ε-sweep + targeted-LAT experiments (critique) |
| `evolvability.py` | Wagner evolvability probe: adapt-to-new-concept + compositionality |
