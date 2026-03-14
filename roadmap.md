# Pantanal Sentinel — Roadmap

## Competition: BirdCLEF+ 2026
## Team: rsh rhc (Kaggle)

---

## Sprint 0: Foundation (Mar 14–21, 2026)

### Goals
- [x] Register on Kaggle and join BirdCLEF+ 2026
- [x] Create design document
- [ ] Complete Kaggle notebook account setup
- [ ] Download competition dataset
- [ ] Run initial EDA on soundscape data
- [ ] Research winning solutions from BirdCLEF 2025
- [ ] Produce technique cards for top approaches
- [ ] Research Perch v2 embeddings
- [ ] Set up local development environment (bird_env, Python 3.12)
- [ ] Build baseline model (single backbone, mel spectrograms)
- [ ] Implement evaluate.py matching Kaggle's macro ROC-AUC
- [ ] Establish first ROC-AUC baseline number
- [ ] Register for CLEF 2026 lab (deadline: Apr 23)

### Deliverables
- `research/technique_cards/` — at least 3 technique cards
- `research/winning_solutions.md` — top solution teardowns
- `eval/evaluate.py` — ROC-AUC scorer
- First entry in `results.tsv`

---

## Sprint 1: Single Backbone Training (Mar 22–Apr 4, 2026)

### Goals
- [ ] Extract Perch v2 embeddings on full training set
- [ ] Train EfficientNet-B0 on mel spectrograms
- [ ] Train EfficientNet-B1 variant
- [ ] Implement basic data augmentation (mixup, time-stretch)
- [ ] Set up cross-validation by recording site
- [ ] Per-species ROC-AUC analysis to identify weak taxa
- [ ] First Kaggle submission (establish leaderboard position)

### Deliverables
- Trained single-backbone model with known ROC-AUC
- `eval/species_report.md` — per-species breakdown
- First leaderboard score

---

## Sprint 2: Ensemble & Pseudo-Labeling (Apr 5–18, 2026)

### Goals
- [ ] Add ConvNeXt backbone
- [ ] Build weighted ensemble pipeline
- [ ] Implement pseudo-labeling Round 0 (train on labeled, predict unlabeled)
- [ ] Begin ecological priors (species_range.json, geographic filtering)
- [ ] Domain-relevant noise injection for augmentation
- [ ] Start inference optimization planning (ONNX export tests)

### Deliverables
- Multi-backbone ensemble with improved ROC-AUC
- First round of pseudo-labels
- `priors/species_range.json` v1

---

## Sprint 3: Iterative Self-Training (Apr 19–May 2, 2026)

### Goals
- [ ] Multi-round pseudo-labeling pipeline (automated)
- [ ] Temporal masking (diurnal/nocturnal priors)
- [ ] SED (Sound Event Detection) mode implementation
- [ ] Domain adaptation techniques
- [ ] Advanced augmentation (pitch-shift, noise profiles from Pantanal)
- [ ] HITL calibration tool for uncertain detections

### Deliverables
- Pseudo-labeling pipeline producing stable ROC-AUC gains
- Full ecological prior stack
- `tools/reviewer.py` — HITL interface

---

## Sprint 4: Inference Hardening (May 3–16, 2026)

### Goals
- [ ] Export final model(s) to ONNX
- [ ] INT8 quantization with ROC-AUC validation
- [ ] CPU inference profiling (must be ≤ 90 min)
- [ ] Kaggle notebook generator (`inference/submit.py`)
- [ ] Test submission on Kaggle (verify it runs end-to-end)
- [ ] Model size optimization (prune if needed)

### Deliverables
- CPU-only inference pipeline verified ≤ 90 min
- Working Kaggle submission notebook
- `inference/profile_results.md`

---

## Sprint 5: Final Push (May 17–27, 2026)

### Goals
- [ ] Final hyperparameter tuning
- [ ] Leaderboard management (strategic submissions)
- [ ] Ensemble weight optimization
- [ ] Submission hardening and edge case testing
- [ ] **ENTRY DEADLINE: May 27, 2026**

### Deliverables
- Final competition submission
- Best possible leaderboard position

---

## Sprint 6: Working Notes (May 28–Jun 17, 2026)

### Goals
- [ ] Draft Working Notes paper (CEUR-WS format)
- [ ] Document methodology, architecture, and results
- [ ] Include ablation studies
- [ ] Figures: architecture diagram, ROC-AUC curves, per-species heatmaps
- [ ] **WORKING NOTES DEADLINE: Jun 17, 2026**
- [ ] Camera-ready revision by Jul 6, 2026

### Deliverables
- `research/working_notes_draft.md` → final PDF
- Submitted to CEUR-WS proceedings

---

## Key Milestones

| Date | Milestone | Status |
|---|---|---|
| Mar 11, 2026 | Competition launched | Done |
| Mar 14, 2026 | Project kickoff | Done |
| Apr 23, 2026 | CLEF registration close | Pending |
| May 27, 2026 | **ENTRY DEADLINE** | Pending |
| Jun 17, 2026 | Working Notes submission | Pending |
| Jul 6, 2026 | Camera-ready deadline | Pending |
| Sep 21–24, 2026 | CLEF 2026 Conference (Jena) | Pending |
