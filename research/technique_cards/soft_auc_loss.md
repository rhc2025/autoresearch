# Technique Card: SoftAUCLoss — Direct AUC Optimization

## Summary
A differentiable approximation of ROC-AUC as a training loss function, using pairwise ranking between positive and negative samples. Directly optimizes the competition metric rather than proxy losses (BCE, CE).

## Source
- Used by multiple top-5 BirdCLEF+ 2025 teams
- Based on pairwise AUC optimization literature

## Properties
- **Directly optimizes ROC-AUC** (the competition metric)
- **Resistant to overfitting** compared to BCE/CE
- **Supports soft labels** natively (works with pseudo-labels)
- **Per-species optimization** — each species gets its own AUC signal

## How It Works

### Pairwise Formulation
For each species, ROC-AUC measures the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample:

```
AUC = P(score(positive) > score(negative))
```

### Differentiable Approximation
```python
import torch
import torch.nn.functional as F

def soft_auc_loss(y_pred, y_true):
    """Soft AUC loss via pairwise log-loss.

    y_pred: (batch, n_species) — raw logits or probabilities
    y_true: (batch, n_species) — binary or soft labels
    """
    losses = []
    for i in range(y_true.shape[1]):  # per species
        pos_mask = y_true[:, i] > 0.5
        neg_mask = y_true[:, i] <= 0.5

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_scores = y_pred[:, i][pos_mask]
        neg_scores = y_pred[:, i][neg_mask]

        # Pairwise differences: each positive vs each negative
        # shape: (n_pos, n_neg)
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        # Soft ranking loss (log-sigmoid of pairwise difference)
        loss = -F.logsigmoid(diff).mean()
        losses.append(loss)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)
```

### With Soft Labels (for pseudo-labeling)
When using pseudo-labels with continuous confidence scores:
```python
# Weight pairwise comparisons by label confidence
pos_weights = y_true[:, i][pos_mask]
neg_weights = 1.0 - y_true[:, i][neg_mask]
pair_weights = pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
loss = -(F.logsigmoid(diff) * pair_weights).sum() / pair_weights.sum()
```

## When to Use

### SoftAUCLoss vs Focal BCE vs CE

| Scenario | Recommended Loss |
|----------|-----------------|
| Early training, single-label data | CE (BirdCLEF 2024 1st place finding) |
| Multi-label training | Focal BCE |
| Fine-tuning with pseudo-labels | SoftAUCLoss |
| Final stage, optimizing metric | SoftAUCLoss |

### Typical Training Schedule
1. **Epochs 1-10**: Train with Focal BCE (faster convergence)
2. **Epochs 11-30**: Switch to SoftAUCLoss (optimize metric directly)

Or use combined loss:
```
total_loss = 0.5 * focal_bce + 0.5 * soft_auc
```

## Risks
- Slower training (O(n_pos × n_neg) per species per batch)
- May need larger batch sizes for stable gradient estimates
- Not all species have enough positives per batch — consider accumulation
- Memory-intensive for species with many positives

## Mitigation
- Sample fixed number of positive/negative pairs per species per batch
- Use gradient accumulation for effective larger batch sizes
- Fall back to Focal BCE for species with < 5 positives in batch

## Priority for Pantanal Sentinel
**MEDIUM** — Implement after baseline is working. Use as fine-tuning loss in Sprint 2-3 combined with pseudo-labeling pipeline.
