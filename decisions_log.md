# Decisions Log — Pantanal Sentinel

## DEC-001: Design Document First
**Date**: 2026-03-14
**Decision**: Write comprehensive design document before any implementation code.
**Rationale**: Competition has multiple interacting systems (training, inference, priors, paper). Need clear architecture before building.
**Alternatives Considered**: Jump straight to baseline model.
**Outcome**: `pantanal_sentinel_design.md` created with full agent architecture, sprint plan, and risk register.

## DEC-002: Multi-Agent Architecture
**Date**: 2026-03-14
**Decision**: Use 8 specialized agents (PM, Research, Training, Data, Eval, Inference, Priors, Code Review) coordinated through shared repo artifacts.
**Rationale**: Competition scope exceeds single-agent capacity. Different phases need different tools (Gemini for research, RTX 5090 for training, CPU profiling for inference).
**Tools Mapped**: Claude Code, Google Gemini, Google AI Studio, Google Antigravity, VS Code, Codex, Open Source LLM.

## DEC-003: Autoresearch Pattern for Training Loop
**Date**: 2026-03-14
**Decision**: Adapt the autoresearch experiment loop (modify → train → eval → keep/revert) for the BirdCLEF training pipeline.
**Rationale**: Proven pattern for autonomous metric optimization. Maps cleanly: val_bpb → ROC-AUC, train.py → models/train.py.
**Key Difference**: Added outer loop for pseudo-labeling and dual constraint (ROC-AUC + CPU inference time).

## DEC-004: Sprint Cadence
**Date**: 2026-03-14
**Decision**: 6 sprints of ~2 weeks each, from Mar 14 through Jun 17.
**Rationale**: Matches competition timeline. Early sprints focus on model quality, later sprints on inference optimization and paper.
**Risk**: If pseudo-labeling takes longer than expected, Sprints 2-3 may compress Sprint 4 (inference).
