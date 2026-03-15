# Status Board — Pantanal Sentinel

**Last Updated**: 2026-03-15
**Current Sprint**: Sprint 0 — Foundation (Mar 14–21)
**Days Until Entry Deadline**: 73

---

## Workstream Status

| Workstream | Agent | Status | Last Update |
|---|---|---|---|
| PM / Orchestration | Claude Code | Active | Sprint 0 tracking, coordination framework built |
| Research | Claude Code | Done (Sprint 0) | 5 technique cards + winning solutions teardown complete |
| Data & Features | **Antigravity** | Awaiting Handoff | `data/dataset.py` scaffolded; needs competition data download + EDA |
| Model Training | Claude Code | Scaffolded | `models/train.py` + EfficientNet backbone ready; blocked on data |
| Evaluation | Claude Code | Scaffolded | `eval/evaluate.py` implemented (macro ROC-AUC) |
| Inference Optimization | Claude Code | Not Started | Sprint 4 focus |
| Ecological Priors | **Antigravity** | Not Started | Sprint 2 focus |
| Working Notes Paper | Claude Code | Not Started | Sprint 6 focus |

### Active Models
| Model | Platform | Last Active | Current Task |
|-------|----------|-------------|-------------|
| Claude Code | Anthropic CLI | 2026-03-15 | Coordination framework, research, scaffolding |
| Antigravity | Google Agentic IDE | — | Onboarding — see `HANDOFF.md` |

---

## Current Best Results

| Metric | Value | Commit | Date |
|---|---|---|---|
| ROC-AUC (macro) | — | — | — |
| Inference Time (CPU) | — | — | — |
| Model Size | — | — | — |

---

## Blockers

| Blocker | Owner | Impact | Resolution |
|---|---|---|---|
| Kaggle notebook setup in progress | User | Cannot submit yet | User working on it |
| Competition data not downloaded | User/Data Agent | Cannot train | Need Kaggle API key or manual download |
| CLEF 2026 lab registration | User | Required for Working Notes | Deadline: Apr 23 |

---

## Recent Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-03-14 | Design doc first, then implement | Proper architecture before code |
| 2026-03-14 | Multi-agent approach | Competition requires PM, Research, Training, Eval, Inference specialization |
| 2026-03-14 | Autoresearch pattern for training | Proven loop: modify → train → eval → keep/revert |

---

## Sprint 0 Progress

- [x] Register on Kaggle
- [x] Join BirdCLEF+ 2026 competition
- [x] Create design document
- [x] Create roadmap and status board
- [x] Research Perch v2 → technique card at `research/technique_cards/perch_v2_embeddings.md`
- [x] Research winning solutions → `research/winning_solutions.md` (5 technique cards)
- [ ] Research competition dataset
- [x] Implement ROC-AUC evaluator → `eval/evaluate.py`
- [x] Scaffold baseline model → `models/train.py`, `models/backbones/efficientnet.py`
- [x] Scaffold dataset pipeline → `data/dataset.py`
- [ ] Download competition data (BLOCKER)
- [ ] Set up local dev environment (bird_env, dependencies)
- [ ] Build and train baseline model (blocked on data)
- [ ] Establish first ROC-AUC baseline
