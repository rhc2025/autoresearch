# Pantanal Sentinel — Agent Constitution

## Project
BirdCLEF+ 2026 competition entry. Identify wildlife vocalizations (birds, amphibians, mammals, reptiles, insects) in Pantanal soundscapes. Maximize macro-averaged ROC-AUC. CPU-only inference ≤ 90 minutes.

## Key Files
- `pantanal_sentinel_design.md` — full architecture and agent design
- `roadmap.md` — sprint plan with deadlines
- `status_board.md` — current progress (update after completing work)
- `decisions_log.md` — architectural decisions
- `program.md` — original autoresearch experiment loop pattern (reference)

## Deadlines
- **May 27, 2026**: Kaggle entry deadline
- **Jun 17, 2026**: Working Notes paper submission
- **Apr 23, 2026**: CLEF 2026 lab registration close

## Rules for All Agents
1. **Update status_board.md** after completing significant work
2. **Log decisions** in decisions_log.md for non-trivial choices
3. **Track experiments** in results.tsv (untracked by git)
4. **Do not modify** evaluation harness once established — it must match Kaggle's scorer
5. **Commit frequently** with descriptive messages
6. **ROC-AUC is the metric** — every change must be measured against it
7. **CPU inference time is the constraint** — models must run in ≤ 90 min on Kaggle CPU

## Directory Structure
```
research/          — technique cards, solution teardowns, paper draft
models/            — training code (the edit surface)
data/              — preprocessing, augmentation, pseudo-labels
eval/              — ROC-AUC scorer, analysis
inference/         — ONNX export, CPU optimization, submission notebook
priors/            — ecological priors (geographic, temporal)
tools/             — HITL reviewer, KML export, SQLite store
agents/            — agent-specific instruction files
```

## Hardware
- GPU: NVIDIA RTX 5090 (24GB GDDR7)
- CPU: Intel Core Ultra 9 285HX
- Training budget: 15-30 min per experiment (configurable)
