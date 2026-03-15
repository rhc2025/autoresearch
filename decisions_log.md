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

## DEC-005: Baseline Architecture — EfficientNet-B0 SED with Attention Pooling
**Date**: 2026-03-15
**Decision**: Use EfficientNet-B0 (timm `tf_efficientnet_b0_ns`) as initial backbone with attention-based pooling SED head and BCEWithLogits loss.
**Rationale**: EfficientNet-B0 is the most common baseline across all BirdCLEF 2024/2025 winners. Fast inference, well-understood, pretrained weights available. Attention pooling (vs. mean pooling) enables the model to focus on the most informative time frames.
**Alternatives Considered**: eca_nfnet_l0 (2nd place 2025), EfficientNetV2-S (heavier). Will add these in Sprint 1 for ensemble diversity.

## DEC-006: Iterative Pseudo-Labeling as Primary Strategy
**Date**: 2026-03-15
**Decision**: Adopt multi-iterative Noisy Student with power scaling as the primary accuracy improvement strategy (from 1st place BirdCLEF+ 2025).
**Rationale**: Single biggest technique: +0.058 AUC in 2025 competition. All top-5 solutions used some form of pseudo-labeling. Plan for 4 rounds starting Sprint 2.
**Risk**: Requires working baseline first + unlabeled soundscape data.

## DEC-007: OpenVINO FP16 for Inference
**Date**: 2026-03-15
**Decision**: Target PyTorch → ONNX → OpenVINO FP16 as the inference pipeline.
**Rationale**: 2nd and 5th place BirdCLEF+ 2025 used this path. FP16 preferred over INT8 (preserves accuracy, winners didn't use INT8). Enables 2-4 model ensemble within 90-min budget.
**Alternative**: ONNX only (BirdCLEF 2024 3rd place noted ~0.01 AUC drop with OpenVINO). Will benchmark both.

## DEC-004: Sprint Cadence
**Date**: 2026-03-14
**Decision**: 6 sprints of ~2 weeks each, from Mar 14 through Jun 17.
**Rationale**: Matches competition timeline. Early sprints focus on model quality, later sprints on inference optimization and paper.
**Risk**: If pseudo-labeling takes longer than expected, Sprints 2-3 may compress Sprint 4 (inference).
