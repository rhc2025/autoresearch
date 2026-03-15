# Handoff Queue — Cross-Model Task Sync

**Last Updated**: 2026-03-15

---

## Pending Handoffs

### [2026-03-15] From: Claude Code → To: Antigravity
**Branch:** `claude/explore-agentic-patterns-oTB7I` (merged to master when ready)
**What:** Scaffolded full training pipeline — models, eval, dataset, configs
**Artifacts:**
- `models/train.py` — training loop (needs data to run)
- `models/backbones/efficientnet.py` — EfficientNet-B0 SED backbone
- `models/configs/baseline.yaml` — mel spectrogram config (128 bands, 5s clips)
- `data/dataset.py` — audio → mel spectrogram pipeline
- `eval/evaluate.py` — macro ROC-AUC scorer
- `research/winning_solutions.md` — top BirdCLEF+ 2025 solutions
- `research/technique_cards/` — 5 technique cards (pseudo-labeling, taxonomic pipelines, CPU inference, SoftAUC, Perch v2)

**Next Steps for Antigravity:**
1. Download competition dataset from Kaggle → `data/raw/`
2. Run initial EDA on soundscape data (species distribution, recording lengths, SNR)
3. Validate that `data/dataset.py` loads audio correctly with real data
4. Begin Perch v2 embedding extraction if model weights are accessible
5. Set up `data/augmentation.py` (MixUp, time-shift, random crop)

**Interface Contract:**
- Dataset returns `(mel_spectrogram: Tensor[1, 128, 313], label: Tensor[n_species])`
- Mel config lives in `models/configs/baseline.yaml` — changes need both models to agree

**Status:** PENDING

---

## Completed Handoffs

_(none yet)_

---

## Notes

- Both models should check this file at the start of every session
- Mark items COMPLETED when you've picked them up and finished
- Add new handoffs at the top of "Pending Handoffs"
