# Contributing — Multi-Model Coordination Protocol

## Overview

This repo is shared between two frontier AI models working as autonomous agents:

| Model | Platform | Primary Role | Branch Prefix |
|-------|----------|-------------|---------------|
| **Claude Code** | Anthropic CLI | PM, Research, Training, Eval, Inference | `claude/` |
| **Antigravity** | Google Agentic IDE | Data & Features, Pseudo-labeling, Embeddings | `antigravity/` |

**Human operator** (rhc) reviews PRs, resolves conflicts, and makes final merge decisions.

---

## 1. Branch Convention

```
master                        ← stable, tested code only
claude/<description>-<id>     ← Claude Code feature branches
antigravity/<description>     ← Antigravity feature branches
```

**Rules:**
- Never push directly to `master`
- Each model works on its own prefixed branches
- PRs to master require human approval (for now — automation later)

---

## 2. Coordination Files (Read Before Working)

Before starting any work, **both models MUST read** these files:

| File | Purpose | Who Updates |
|------|---------|-------------|
| `status_board.md` | Current state of all workstreams | Both — after completing work |
| `decisions_log.md` | Architectural decisions with rationale | Both — for non-trivial choices |
| `roadmap.md` | Sprint plan, what's done, what's next | Claude Code (PM role) |
| `HANDOFF.md` | Cross-model task queue and sync state | Both — before/after handoffs |
| `results.tsv` | Experiment metrics (untracked by git) | Both — after experiments |

### Update Protocol
1. `git pull origin master` before starting any work session
2. Read `status_board.md` and `HANDOFF.md`
3. Do your work on your own branch
4. Update `status_board.md` with what you did
5. Update `HANDOFF.md` if you have tasks/artifacts for the other model
6. Commit, push, open PR

---

## 3. Directory Ownership

Each model has **primary ownership** of certain directories. The owner makes most changes; the other model may read freely but should coordinate before editing.

| Directory | Primary Owner | Description |
|-----------|--------------|-------------|
| `research/` | Claude Code | Technique cards, solution teardowns, paper |
| `models/` | Claude Code | Training code, backbones, ensemble |
| `eval/` | Claude Code | ROC-AUC scorer, analysis |
| `inference/` | Claude Code | ONNX export, CPU optimization |
| `data/` | Antigravity | Preprocessing, augmentation, pseudo-labels |
| `priors/` | Antigravity | Geographic/temporal ecological priors |
| `tools/` | Antigravity | HITL reviewer, KML export, SQLite |
| `agents/` | Both | Agent-specific instruction files |

**Shared files** (both may edit, coordinate via HANDOFF.md):
- `status_board.md`, `decisions_log.md`, `roadmap.md`, `HANDOFF.md`
- `CLAUDE.md` (project instructions)
- `pyproject.toml`, `uv.lock` (dependencies)

---

## 4. Handoff Protocol

When one model produces output that the other model needs:

### Sender (model that finished work)
1. Push your branch and open a PR (or note the branch name)
2. Add an entry to `HANDOFF.md` under `## Pending Handoffs`:
   ```markdown
   ### [YYYY-MM-DD] From: <sender> To: <receiver>
   **Branch:** <branch-name>
   **What:** Brief description of what was done
   **Artifacts:** List of files created/modified
   **Next Steps:** What the receiver should do with this
   **Status:** PENDING
   ```

### Receiver (model picking up work)
1. Read `HANDOFF.md` — check for PENDING items addressed to you
2. Pull the relevant branch or merged master
3. Do your work
4. Update the handoff entry status to `COMPLETED`
5. Add your own handoff if there's follow-up for the sender

---

## 5. Decision-Making

### Who decides what?
- **Within your owned directories**: You decide. Log in `decisions_log.md`.
- **Shared interfaces** (e.g., data format between `data/` and `models/`): Propose in `HANDOFF.md`, wait for acknowledgment or human approval.
- **Architecture-level changes**: Log in `decisions_log.md` with rationale. First to log wins unless the human overrides.

### Conflict Resolution
1. Check `decisions_log.md` — if a decision was already made, follow it
2. If models disagree, the human operator arbitrates
3. Never silently override another model's decision

---

## 6. Commit Messages

Both models should follow this format:

```
<type>(<scope>): <short description>

<body — what and why>

Co-authored-by: <model-name>
```

**Types:** feat, fix, refactor, research, data, eval, infra, docs
**Scopes:** models, data, eval, inference, priors, research, tools

Example:
```
feat(data): Add mel spectrogram preprocessing pipeline

Implements 128-band log-mel spectrograms with configurable
f_min/f_max for taxonomic-specific frequency ranges.
Outputs HDF5 for fast training IO.

Co-authored-by: Antigravity
```

---

## 7. Shared Interfaces

These are the contracts between Claude Code's models and Antigravity's data:

### Training Data Format
```python
# data/dataset.py defines the interface:
# - Input: audio file path + metadata
# - Output: (mel_spectrogram: Tensor[1, n_mels, time], label: Tensor[n_species])
# - Mel config in models/configs/baseline.yaml
```

### Pseudo-Label Format
```
data/pseudo_labels/round_N.csv
Columns: filename, species_id, confidence, round
```

### Experiment Logging
```
results.tsv
Columns: timestamp, experiment_name, model, auc_macro, auc_per_species_json,
         training_time_sec, inference_time_sec, commit_hash, notes
```

---

## 8. PR Automation (Progressive)

### Phase 1 (Now): Manual PRs
- Each model pushes to its branch
- Human reviews and merges via GitHub

### Phase 2 (Sprint 1+): Semi-automated
- PRs auto-created on push via GitHub Actions
- Lint + eval checks run automatically
- Human approves merge

### Phase 3 (Sprint 3+): Automated merge for safe changes
- Auto-merge PRs that:
  - Only touch owned directories
  - Pass all CI checks (lint, eval, tests)
  - Don't modify shared interfaces
- Human approval still required for:
  - Cross-directory changes
  - Interface changes
  - Dependency updates

---

## 9. Getting Started (for Antigravity)

If you're Antigravity starting a new session:

1. `git pull origin master`
2. Read: `CLAUDE.md`, `status_board.md`, `HANDOFF.md`, `decisions_log.md`
3. Check `roadmap.md` for current sprint priorities
4. Create branch: `antigravity/<descriptive-name>`
5. Work in your owned directories (`data/`, `priors/`, `tools/`)
6. Update `status_board.md` and `HANDOFF.md` when done
7. Push and note the branch name in `HANDOFF.md`

### Key Context Files
- `pantanal_sentinel_design.md` — full architecture
- `research/winning_solutions.md` — what top teams did
- `research/technique_cards/` — prioritized techniques
- `models/configs/baseline.yaml` — current model config (shared interface)
