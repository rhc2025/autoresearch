# Technique Card: Taxonomic-Specific Pipelines

## Summary
Train separate model pipelines for different taxonomic groups (birds vs. insects/amphibians/mammals) because their acoustic signatures occupy fundamentally different frequency bands and temporal patterns.

## Source
- 1st Place BirdCLEF+ 2025 (Nikita Babych)

## Impact
- **+0.003 AUC** on top of an already strong 0.930 baseline
- Small but consistent gain; low implementation cost

## How It Works

### Problem
BirdCLEF+ is multi-taxa: birds, amphibians, mammals, reptiles, insects. These groups differ in:
- **Frequency range**: Bird songs 1-10 kHz; insects 3-20 kHz; frogs 0.5-5 kHz
- **Temporal patterns**: Birds have distinct syllables; insects produce continuous trills; frogs repeat at fixed intervals
- **Signal-to-noise**: Insect sounds are often broadband and harder to isolate

A single model trained on all taxa simultaneously may not learn optimal features for each group.

### Solution
```
Pipeline 1: Birds + mammals → EfficientNet ensemble (main pipeline)
Pipeline 2: Insects + amphibians → EfficientNet-B0 (specialized)
Final: Merge predictions by taxonomic group
```

### Merging Strategy
- Each pipeline produces probabilities for its own species set
- Final submission concatenates predictions from both pipelines
- No cross-pipeline calibration needed if trained on same data splits

## Implementation Notes

### Species Classification
- Use taxonomy metadata from competition to split species into groups
- Pantanal species list will include ~150+ birds, ~20 amphibians, ~15 mammals, ~10 insects, ~5 reptiles
- Group 1: Birds + mammals (~170 species)
- Group 2: Amphibians + insects + reptiles (~36 species)

### Audio Preprocessing
- Pipeline 1 (birds): Standard mel spectrogram, f_min=300 Hz, f_max=12000 Hz
- Pipeline 2 (non-birds): Wider range, f_min=60 Hz, f_max=16000 Hz
- Consider different hop lengths for temporal resolution

### Training
- Both pipelines follow same training loop (same augmentation, pseudo-labeling)
- Pipeline 2 may need more aggressive oversampling (fewer samples per species)

## Risks
- Adds inference time (~15-20% more due to second model)
- Must fit within 90-min CPU budget
- Marginal gain may not justify complexity in early sprints

## Priority for Pantanal Sentinel
**MEDIUM** — Implement after pseudo-labeling pipeline is working (Sprint 3). The Pantanal dataset is also multi-taxa, so this technique directly applies.
