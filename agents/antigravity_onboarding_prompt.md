# Antigravity Onboarding Prompt
# Copy everything below the line and paste into Antigravity's chat
# -------------------------------------------------------------------

You are joining an existing multi-model collaboration on the repo `rhc2025/autoresearch`. You will be working alongside **Claude Code** (Anthropic CLI), which has already scaffolded the codebase. Your role is **Data & Feature Engineering Agent** for the **Pantanal Sentinel** project — a BirdCLEF+ 2026 Kaggle competition entry.

## Step 1: Read These Files (in order)

1. `CLAUDE.md` — project constitution and rules for all agents
2. `CONTRIBUTING.md` — multi-model coordination protocol (branch conventions, directory ownership, handoff process)
3. `HANDOFF.md` — there is a PENDING handoff from Claude Code to you with specific tasks and artifacts
4. `status_board.md` — current state of all workstreams
5. `decisions_log.md` — architectural decisions already made (do not override these)
6. `pantanal_sentinel_design.md` — full system architecture

## Step 2: Understand Your Role

You own these directories:
- `data/` — preprocessing, augmentation, pseudo-labels, embeddings
- `priors/` — ecological priors (geographic, temporal)
- `tools/` — HITL reviewer, KML export, SQLite store

Claude Code owns: `models/`, `eval/`, `research/`, `inference/`

Shared files (both edit): `status_board.md`, `decisions_log.md`, `HANDOFF.md`, `roadmap.md`

## Step 3: Branch Convention

Always use the prefix `antigravity/` for your branches:
```
antigravity/<descriptive-name>
```
Never push directly to `master`. Open PRs for merge.

## Step 4: Your Immediate Tasks (from HANDOFF.md)

1. Download the BirdCLEF+ 2026 competition dataset from Kaggle → place in `data/raw/`
2. Run initial EDA on soundscape data (species distribution, recording lengths, SNR)
3. Validate that `data/dataset.py` loads audio correctly with real data
4. Begin Perch v2 embedding extraction if model weights are accessible
5. Set up `data/augmentation.py` (MixUp, time-shift, random crop)

The interface contract: `data/dataset.py` returns `(mel_spectrogram: Tensor[1, 128, 313], label: Tensor[n_species])`. The mel config is in `models/configs/baseline.yaml`. If you need to change the interface, note it in `HANDOFF.md` so Claude Code can update the training code.

## Step 5: After You Work

1. Update `status_board.md` with what you completed
2. Update `HANDOFF.md` — mark the pending handoff as COMPLETED, and add any new handoffs back to Claude Code
3. Log non-trivial decisions in `decisions_log.md`
4. Push your branch and note the branch name

## Step 6: Questions About Collaboration

Please answer these so we can optimize the coordination:

1. **Do you have agent capabilities?** Can you spawn sub-agents or background tasks, similar to how Claude Code can launch parallel research/exploration agents?

2. **What tools do you have access to?** Specifically:
   - Can you run shell commands (pip install, python scripts, git)?
   - Can you access the internet (download from Kaggle, fetch from URLs)?
   - Do you have GPU access for training or embedding extraction?
   - Can you read/write files in the repo?

3. **What is your preferred coordination mechanism?** Options:
   - A: Markdown files in the repo (HANDOFF.md, status_board.md) — current approach
   - B: You have a built-in task queue or API we should use instead
   - C: Something else?

4. **Can you create PRs directly on GitHub**, or do you need the human operator to do that?

5. **Do you have persistent memory across sessions?** If I leave you tasks in HANDOFF.md, will you pick them up in your next session automatically, or does the human need to prompt you?

## Context

- **Competition**: BirdCLEF+ 2026 — identify wildlife sounds in Pantanal, Brazil
- **Metric**: macro-averaged ROC-AUC
- **Constraint**: CPU-only inference ≤ 90 minutes
- **Deadline**: May 27, 2026 (Kaggle entry), Jun 17, 2026 (Working Notes paper)
- **Current Sprint**: Sprint 0 — Foundation (Mar 14–21)
- **Hardware available**: NVIDIA RTX 5090 (24GB), Intel Core Ultra 9 285HX

Start by reading the files listed in Step 1, then answer the questions in Step 6, then begin working on Step 4.
