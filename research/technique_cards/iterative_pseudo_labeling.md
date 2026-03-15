# Technique Card: Multi-Iterative Noisy Student with Power Scaling

## Summary
Run multiple rounds (4+) of pseudo-labeling where each round trains a fresh student model on original labels + refined pseudo-labels from the previous teacher. PowerTransform is applied to pseudo-label confidence scores between rounds to prevent label collapse and enable progressive refinement.

## Source
- 1st Place BirdCLEF+ 2025 (Nikita Babych)
- [Kaggle writeup](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)

## Impact
- **+0.058 AUC** (0.872 → 0.930) in BirdCLEF+ 2025
- Single biggest technique in the winning solution

## How It Works

### Basic Pipeline
```
Round 0: Train teacher on labeled data only
Round 1: Teacher predicts unlabeled soundscapes → filter by confidence → train new student
Round 2: Student becomes teacher → predict again → filter → train new student
...
Round N: Repeat until ROC-AUC stabilizes (typically 4 rounds)
```

### Power Scaling (Key Innovation)
Between rounds, apply PowerTransform to pseudo-label probabilities:
```
p_refined = p_raw ^ gamma
```
Where `gamma` controls label sharpening. This:
- Prevents confident predictions from saturating at 1.0
- Allows progressive refinement across rounds
- Avoids the label collapse problem where errors amplify

### Noisy Student Component
Each student model is trained with:
- Stronger augmentation than the teacher (MixUp, StochasticDepth)
- Dropout/noise injection
- This regularization prevents the student from simply memorizing teacher predictions

## Implementation Notes

### Confidence Thresholding
- Round 1: Use higher threshold (e.g., 0.8) to minimize false pseudo-labels
- Later rounds: Can lower threshold as pseudo-labels improve
- Species with very few labeled examples benefit most from pseudo-labels

### Ensemble Teachers
- Use an ensemble of models as the teacher for more reliable pseudo-labels
- Each student can be a single model (cheaper to train)

### Compute Budget
- Each round requires full training (~15-30 min on RTX 5090)
- 4 rounds = 1-2 hours total training time
- Pseudo-label generation on soundscapes: ~10-20 min per round

## Risks
- Error amplification if confidence thresholds are too low
- Overfitting to domain-specific noise patterns in soundscapes
- Diminishing returns after 4-5 rounds
- No reliable local validation in BirdCLEF+ 2025 (relied on public LB)

## Dependencies
- Labeled training data (Xeno-Canto, competition train_audio)
- Unlabeled soundscape data (competition test set or similar)
- Base model capable of reasonable predictions (>0.85 AUC)

## Priority for Pantanal Sentinel
**CRITICAL** — This is the single most impactful technique from BirdCLEF+ 2025 winners. Implement in Sprint 2-3.
