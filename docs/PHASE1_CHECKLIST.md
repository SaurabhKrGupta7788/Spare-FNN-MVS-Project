# Phase 1 run checklist

- [ ] Create and activate a Python 3.10+ environment.
- [ ] Install the appropriate PyTorch build for the machine.
- [ ] Install the project with `pip install -e ".[dev]"`.
- [ ] Run `python scripts/check_environment.py` and confirm PyTorch/Transformers/Datasets imports.
- [ ] Run `python -m sparse_ffn prepare-data` and confirm train, validation and test blocks are saved.
- [ ] Run the pilot extraction with the default 20 batches.
- [ ] Check the extraction `manifest.json` has `status: complete` and the expected 12 layers.
- [ ] Run descriptive analysis.
- [ ] Inspect `layer_summary.csv` and all generated plots.
- [ ] Increase `max_batches` only after checking runtime, memory and disk usage. Float16 raw storage is approximately `tokens × 12 × 3,072 × 2 bytes` before compression.
- [ ] Preserve the extraction manifest and exact model/dataset revisions for reproducibility.

## Before presenting numerical results

- [ ] Report how many token positions and documents/chunks were sampled.
- [ ] Report the activation point exactly: post-ReLU, pre-`fc2`.
- [ ] Report storage precision and sampling settings.
- [ ] Do not treat a pilot sample as final evidence.
- [ ] Do not interpret descriptive differences as statistically significant.
