# Status Board — Pantanal Sentinel

**Last Updated**: 2026-03-14
**Current Sprint**: Sprint 0 — Foundation (Mar 14–21)
**Days Until Entry Deadline**: 74

---

## Workstream Status

| Workstream | Agent | Status | Last Update |
|---|---|---|---|
| PM / Orchestration | Claude Code | Active | Roadmap and status board created |
| Research | Gemini / Claude | In Progress | Researching BirdCLEF 2025 winning solutions |
| Data & Features | — | Not Started | Waiting on Kaggle data download |
| Model Training | — | Not Started | Blocked on baseline setup |
| Evaluation | — | Not Started | Need evaluate.py |
| Inference Optimization | — | Not Started | Sprint 4 focus |
| Ecological Priors | — | Not Started | Sprint 2 focus |
| Working Notes Paper | — | Not Started | Sprint 6 focus |

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
- [ ] Research winning solutions (in progress)
- [ ] Research Perch v2 (in progress)
- [ ] Research competition dataset (in progress)
- [ ] Download competition data
- [ ] Set up local dev environment
- [ ] Build baseline model
- [ ] Implement ROC-AUC evaluator
