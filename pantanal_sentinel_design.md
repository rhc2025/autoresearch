# Pantanal Sentinel — Multi-Agent Design Document

## 1. Mission Statement

Build an autonomous, multi-agent system to compete in **BirdCLEF+ 2026** — identifying wildlife vocalizations (birds, amphibians, mammals, reptiles, insects) in real-world soundscape recordings from the Brazilian Pantanal. The system must maximize **macro-averaged ROC-AUC** over species presence in 5-second intervals, with final submission running **CPU-only in ≤90 minutes** on a Kaggle notebook.

---

## 2. Critical Path & Deadlines

| Milestone | Date | Owner | Notes |
|---|---|---|---|
| Competition launched | Mar 11, 2026 | — | Data available on Kaggle |
| **TODAY** | **Mar 14, 2026** | — | Day 3 of competition |
| CLEF Registration close | Apr 23, 2026 | PM Agent | Must register for CLEF 2026 lab |
| **Entry Deadline** | **May 27, 2026** | PM Agent | Final submission on Kaggle |
| Working Notes Submission | Jun 17, 2026 | Research Agent | Paper describing methods → CEUR-WS |
| Notification of Acceptance | Jun 24, 2026 | — | |
| Camera-Ready Notes | Jul 6, 2026 | Research Agent | Final paper revision |
| CLEF 2026 Conference | Sep 21–24, 2026 | — | Jena, Germany |

### Sprint Cadence (suggested)

| Sprint | Dates | Focus |
|---|---|---|
| Sprint 0 | Mar 14–21 | Setup, EDA, baseline model, agent scaffold |
| Sprint 1 | Mar 22–Apr 4 | Perch v2 embeddings, single-backbone training |
| Sprint 2 | Apr 5–18 | Ensemble, pseudo-labeling pipeline, ecological priors |
| Sprint 3 | Apr 19–May 2 | Iterative self-training, domain adaptation |
| Sprint 4 | May 3–16 | Inference optimization (ONNX/CPU), submission hardening |
| Sprint 5 | May 17–27 | Final tuning, submission, leaderboard management |
| Sprint 6 | May 28–Jun 17 | Working Notes paper writing |

---

## 3. Agent Architecture

### 3.1 PM Agent (Orchestrator)

**Tools**: Claude Code, Codex
**Responsibilities**:
- Owns the roadmap, sprint backlog, and critical path
- Reviews outputs from all other agents
- Tracks deadlines (CLEF registration, entry, working notes)
- Maintains `status_board.md` — a living document of progress
- Assigns priorities and resolves blockers
- Generates weekly status reports

**Loop**:
```
Check deadlines → Review agent outputs → Update priorities → Assign next tasks → Report
```

**Key Artifacts**:
- `roadmap.md` — sprint plan with milestones
- `status_board.md` — current state of all workstreams
- `decisions_log.md` — architectural decisions with rationale

### 3.2 Research Agent

**Tools**: Google Gemini, Google AI Studio, Web Search
**Responsibilities**:
- Analyze winning solutions from BirdCLEF 2024/2025
- Summarize relevant papers (BioME, State Space Models, NBM dataset)
- Scout new techniques from arxiv, Kaggle discussions
- Produce "technique cards" — structured summaries of applicable methods
- Draft the Working Notes paper (CEUR-WS format)

**Loop**:
```
Identify knowledge gap → Search/Read papers → Produce technique card → Recommend to PM
```

**Key Artifacts**:
- `research/technique_cards/` — one file per technique
- `research/winning_solutions.md` — teardowns of top BirdCLEF solutions
- `research/working_notes_draft.md` — the CLEF paper

**Priority Research Targets**:
1. Dylan Liu (4th place 2025) — pseudo-labeling strategy
2. MYSO (5th place 2025) — SED mode stability
3. Nikita Babych — domain-relevant noise injection
4. BioME paper — resource-efficient bioacoustic foundation models
5. Shreejith SG (UCSD) — Pantanal-specific species coverage
6. Perch v2 architecture and embedding extraction

### 3.3 Model Training Agent

**Tools**: Claude Code, VS Code, RTX 5090 (24GB GDDR7)
**Responsibilities**:
- The core autoresearch loop — modify model → train → eval → keep/revert
- Manage the training pipeline for all backbone architectures
- Hyperparameter tuning and architecture search
- Multi-backbone ensemble training (EfficientNet-B0/B1, ConvNeXt)

**Loop** (adapted from autoresearch `program.md`):
```
LOOP FOREVER:
1. Read current state (git log, results.tsv, best ROC-AUC)
2. Propose experiment (architecture change, hyperparameter, augmentation)
3. Modify training code
4. git commit
5. Run training: python train.py > run.log 2>&1
6. Extract metrics: grep "roc_auc\|val_loss" run.log
7. Log to results.tsv
8. If ROC-AUC improved → keep commit
9. If ROC-AUC same/worse → git reset to previous best
```

**Key Artifacts**:
- `models/train.py` — the primary edit surface
- `models/ensemble.py` — multi-backbone ensemble
- `results.tsv` — experiment log (untracked)

**Hardware Budget**:
- RTX 5090: 24GB GDDR7, used for all GPU training
- Training time budget per experiment: configurable (suggest 15-30 min for initial, scale up for promising directions)
- VRAM management: bf16 training, gradient checkpointing as needed

### 3.4 Data & Feature Engineering Agent

**Tools**: Google Antigravity (Agentic IDE), Claude Code
**Responsibilities**:
- Perch v2 (BirdVocalizationClassifier) embedding extraction
- Audio preprocessing pipeline (mel spectrograms, bandpass filtering)
- Data augmentation (mixup, time-stretch, pitch-shift, noise injection)
- Pseudo-labeling pipeline (the competition "breakthrough")
- Dataset management (Xeno-canto, competition soundscapes, unlabeled data)

**Loop**:
```
Extract features → Generate pseudo-labels → Filter by confidence → Augment training set → Retrain
```

**Pseudo-Labeling Pipeline** (critical path):
```
Round 0: Train on labeled Xeno-canto data
Round 1: Predict on unlabeled soundscapes → filter top-K confident → add to training
Round 2: Retrain with expanded dataset → predict again → filter → add
Round N: Repeat until ROC-AUC stabilizes
```

**Key Artifacts**:
- `data/embeddings/` — Perch v2 features (cached)
- `data/pseudo_labels/` — self-training outputs per round
- `data/augmentation.py` — augmentation pipeline
- `data/prepare.py` — data loading and preprocessing

### 3.5 Evaluation & Analytics Agent

**Tools**: Claude Code
**Responsibilities**:
- Compute macro-averaged ROC-AUC (the competition metric)
- Per-species performance analysis (find weak spots)
- Confusion matrix generation
- Experiment tracking and visualization
- Leaderboard position monitoring
- Cross-validation strategy

**Loop**:
```
Run eval → Analyze per-species → Identify weak taxa → Report to PM → Recommend focus areas
```

**Key Artifacts**:
- `eval/evaluate.py` — ROC-AUC scorer matching Kaggle's implementation
- `eval/analysis.ipynb` — results visualization
- `eval/results.tsv` — experiment history
- `eval/species_report.md` — per-species performance breakdown

### 3.6 Inference Optimization Agent

**Tools**: Claude Code, Open Source LLM tools
**Responsibilities**:
- CPU-only inference pipeline (the submission constraint)
- Model export: PyTorch → ONNX → OpenVINO/TFLite
- Quantization (INT8, dynamic quantization)
- Runtime profiling on CPU
- Kaggle notebook generation and testing
- Ensure total inference ≤ 90 minutes on Kaggle CPU

**Loop**:
```
Export model → Quantize → Profile on CPU → Check runtime ≤ 90min → Check ROC-AUC holds → Keep/Revert
```

**Key Artifacts**:
- `inference/submit.py` — Kaggle notebook generator
- `inference/optimize.py` — ONNX/quantization pipeline
- `inference/profile_results.md` — CPU timing benchmarks

### 3.7 Ecological Priors ("Delinting") Agent

**Tools**: Claude Code
**Responsibilities**:
- Geographic constraint filtering (species range maps)
- Temporal masking (diurnal/nocturnal, seasonal)
- Bayesian soft-prior application
- Species range data management
- HITL calibration tool (Platt scaling, isotonic regression)

**Key Artifacts**:
- `priors/species_range.json` — geographic range data
- `priors/temporal_masks.json` — activity curves by species
- `priors/delinting.py` — prior application pipeline
- `tools/reviewer.py` — HITL verification interface

### 3.8 Code Review & Quality Agent

**Tools**: Claude Code
**Responsibilities**:
- Review all PRs before merge
- Enforce code quality standards
- Run test suites
- Check for regressions
- Validate submission notebook integrity

---

## 4. Communication Protocol

### Inter-Agent Communication

Agents communicate through **shared artifacts** (files in the repo) and **the PM Agent**:

```
┌──────────────────────────────────────────────────────────────┐
│                        SHARED REPO                           │
│                                                              │
│  status_board.md  ←── PM reads/writes                        │
│  results.tsv      ←── Training Agent writes, Eval reads      │
│  technique_cards/  ←── Research writes, Training reads        │
│  species_report.md ←── Eval writes, Priors/Training read      │
│  decisions_log.md  ←── PM writes, all read                    │
└──────────────────────────────────────────────────────────────┘
```

### Status Board Format

```markdown
## Status Board — Updated: YYYY-MM-DD

### Current Sprint: Sprint N (dates)

| Workstream | Status | Blocker? | Last Update |
|---|---|---|---|
| Research | 🟢 On Track | — | technique card for pseudo-labeling complete |
| Training | 🟡 In Progress | Need more pseudo-labels | ROC-AUC at 0.82 |
| Inference | 🔴 Blocked | Model not finalized | Waiting on Sprint 3 |
| Priors | 🟢 On Track | — | species_range.json v2 loaded |

### Best ROC-AUC: 0.XXXX (commit abcdef1)
### Days Until Entry Deadline: NN
```

---

## 5. Technology Stack

### Core Stack
- **Python 3.12** — primary language
- **PyTorch** — model training (GPU)
- **ONNX Runtime** — CPU inference
- **librosa / torchaudio** — audio processing
- **Google Perch v2** — foundational embeddings
- **timm** — EfficientNet/ConvNeXt backbones
- **SQLite** — detection database (`bird_library.db`)

### Agent Tooling
| Tool | Role |
|---|---|
| Claude Code | PM, Training, Eval, Inference, Code Review agents |
| Google Antigravity | Data/Feature agent (agentic IDE orchestration) |
| Google AI Studio + Gemini | Research agent (paper analysis, technique scouting) |
| VS Code | Interactive development, debugging |
| Codex | PM assistance, code generation |
| Open Source LLM | Inference optimization, edge case handling |

### Infrastructure
- **GPU**: NVIDIA RTX 5090 (24GB GDDR7) on MSI Raider 18 HX
- **CPU**: Intel Core Ultra 9 285HX
- **Storage**: Local SSD for data, SQLite for detections
- **Version Control**: Git (this repo)
- **Experiment Tracking**: `results.tsv` + `eval/analysis.ipynb`

---

## 6. Project Directory Structure

```
pantanal-sentinel/
├── agents/
│   ├── orchestrator.md          # PM agent instructions
│   ├── research_agent.md        # Research agent instructions
│   ├── training_agent.md        # Training agent instructions (autoresearch-style)
│   ├── data_agent.md            # Data/feature engineering instructions
│   ├── eval_agent.md            # Evaluation agent instructions
│   ├── inference_agent.md       # Inference optimization instructions
│   └── priors_agent.md          # Ecological priors instructions
├── data/
│   ├── raw/                     # Competition soundscapes (gitignored)
│   ├── xeno_canto/              # Training audio (gitignored)
│   ├── embeddings/              # Perch v2 features (gitignored)
│   ├── pseudo_labels/           # Self-training outputs
│   ├── prepare.py               # Data loading and preprocessing
│   └── augmentation.py          # Audio augmentation pipeline
├── models/
│   ├── train.py                 # Main training loop (THE edit surface)
│   ├── ensemble.py              # Multi-backbone ensemble
│   ├── backbones/
│   │   ├── efficientnet.py      # EfficientNet-B0/B1 wrapper
│   │   └── convnext.py          # ConvNeXt wrapper
│   └── configs/                 # Hyperparameter configs per experiment
├── priors/
│   ├── delinting.py             # Ecological prior application
│   ├── species_range.json       # Geographic range data
│   └── temporal_masks.json      # Activity curves
├── inference/
│   ├── submit.py                # Kaggle notebook generator
│   ├── optimize.py              # ONNX/quantization pipeline
│   └── profile_results.md       # CPU timing benchmarks
├── eval/
│   ├── evaluate.py              # ROC-AUC scorer (matches Kaggle)
│   ├── analysis.ipynb           # Results visualization
│   ├── results.tsv              # Experiment log (untracked)
│   └── species_report.md        # Per-species breakdown
├── research/
│   ├── technique_cards/         # One file per technique
│   ├── winning_solutions.md     # Top solution teardowns
│   └── working_notes_draft.md   # CLEF paper draft
├── tools/
│   ├── reviewer.py              # HITL verification tool
│   ├── kml_export.py            # Google Earth visualization
│   └── bird_library.db          # SQLite detection store
├── status_board.md              # Living progress document
├── roadmap.md                   # Sprint plan
├── decisions_log.md             # Architectural decisions
├── results.tsv                  # Global experiment log (untracked)
└── CLAUDE.md                    # Agent constitution / system prompt
```

---

## 7. The Autoresearch Adaptation

The core autoresearch pattern — **constrained edit surface + measurable metric + autonomous loop** — maps directly to this competition:

| Autoresearch Concept | Pantanal Sentinel Equivalent |
|---|---|
| Edit surface: `train.py` | Edit surface: `models/train.py` + `models/ensemble.py` |
| Metric: `val_bpb` (lower = better) | Metric: macro ROC-AUC (higher = better) |
| Time budget: 5 min/experiment | Time budget: 15-30 min/experiment (configurable) |
| Hardware: single GPU | Hardware: RTX 5090 (24GB) |
| Loop: modify → train → eval → keep/revert | Same loop, plus pseudo-labeling outer loop |
| No internet during eval | No internet during Kaggle submission |

### Key Differences from Base Autoresearch

1. **Multiple edit surfaces**: Not just one `train.py` — the ensemble, augmentation, and priors code are all tunable
2. **Outer loop**: Pseudo-labeling creates an outer loop around the training loop
3. **Submission constraint**: Must also optimize for CPU inference time, not just accuracy
4. **Multi-modal evaluation**: ROC-AUC + inference time + per-species analysis
5. **Paper deliverable**: Working Notes paper is a first-class output

---

## 8. Evaluation Strategy

### Primary Metric
- **Macro-averaged ROC-AUC** over species presence in 5-second intervals
- This is the Kaggle leaderboard metric

### Secondary Metrics (internal)
- Per-species ROC-AUC (identify weak taxa for targeted improvement)
- Inference time on CPU (must be ≤ 90 minutes total)
- Model size (smaller = faster inference)
- Pseudo-label quality (precision of confident predictions)

### Cross-Validation
- Stratified K-fold by recording site and species frequency
- Hold out specific sites for geographic generalization testing
- Time-based splits for temporal robustness

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Inference exceeds 90 min on CPU | Cannot submit | Start inference optimization in Sprint 2, not Sprint 4 |
| Domain shift (Xeno-canto → soundscapes) | Low ROC-AUC | Pseudo-labeling + domain-relevant noise injection |
| Overfitting to known species | Poor generalization | Ecological priors + cross-validation by site |
| CLEF registration missed | Cannot submit Working Notes | PM Agent tracks Apr 23 deadline |
| GPU OOM on 24GB | Training blocked | Gradient checkpointing, mixed precision, smaller batch |
| Pseudo-labels amplify errors | ROC-AUC plateau | Confidence thresholding, iterative refinement |

---

## 10. Success Criteria

| Level | Target |
|---|---|
| **Minimum Viable** | Valid Kaggle submission that runs in ≤ 90 min, Working Notes submitted |
| **Competitive** | Top 25% on leaderboard, ROC-AUC > 0.80 |
| **Prize Contention** | Top 5 on leaderboard, ROC-AUC > 0.88 |
| **Best Paper** | Working Notes wins $2,500 prize |

---

## 11. Immediate Next Steps (Sprint 0)

1. **PM Agent**: Set up roadmap.md, status_board.md, register for CLEF 2026
2. **Research Agent**: Tear down Dylan Liu's 4th place 2025 solution, produce first technique card
3. **Data Agent**: Download competition data, run EDA, extract Perch v2 embeddings on a sample
4. **Training Agent**: Build baseline model (single EfficientNet-B0 on mel spectrograms), establish first ROC-AUC baseline
5. **Eval Agent**: Implement `evaluate.py` matching Kaggle's macro ROC-AUC exactly
6. **All**: Validate the repo structure and agent communication protocol

---

## 12. Prize Structure

| Award | Amount |
|---|---|
| 1st Place | $12,000 |
| 2nd Place | $10,000 |
| 3rd Place | $8,000 |
| 4th Place | $5,000 |
| 5th Place | $5,000 |
| 6th–10th Place | $1,000 each |
| Best Working Notes (×2) | $2,500 each |
| **Total** | **$50,000** |

---

## 13. References

- [BirdCLEF+ 2026 (Kaggle)](https://www.kaggle.com/competitions/birdclef-2026)
- [BirdCLEF++ (LifeCLEF/ImageCLEF)](https://www.imageclef.org/BirdCLEF2026)
- [LifeCLEF 2026](https://www.imageclef.org/LifeCLEF2026)
- [CLEF 2026 Conference](https://clef2026.clef-initiative.eu/)
- [Dylan Liu — 4th Place BirdCLEF 2025](https://github.com/dylanliu2002/birdclef-2025)
- [BioME Paper](https://arxiv.org/abs/2307.12058)
- [Perch v2 (Google)](https://github.com/google-research/perch)
