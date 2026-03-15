# Technique Card: Perch v2 Embeddings

## Summary
Google Perch v2 (BirdVocalizationClassifier) is an EfficientNet-B3 backbone (~12M params) trained on 1.5M+ recordings from Xeno-Canto, iNaturalist, Animal Sound Archive, and FSD50K. Produces 1536-dimensional embeddings that are linearly separable for downstream classification.

## Key Specs

| Property | Value |
|---|---|
| Architecture | EfficientNet-B3 (~12M params) |
| Embedding dim | 1536 |
| Label space | ~15,000 classes (~10K birds + amphibians, mammals, insects) |
| Input | 5-sec mono audio @ 32 kHz (160,000 samples) |
| Frontend | Log mel-spectrogram: 128 mel bins, 60 Hz–16 kHz, 10ms hop, 20ms window |
| Outputs | Mean embedding (1536-d), Spatial embedding (5×3×1536), Logits (~15K) |
| Kaggle model | `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1` |
| Also on | Hugging Face: `cgeorgiaw/Perch` |

## Training Innovations in v2
- Self-distillation via ProtoPNet (prototype-learning) generating soft targets
- Multi-source mixup (>2 sources mixed)
- Source-recording prediction as auxiliary task
- Multi-taxa training (not just birds — harder classification = better embeddings)
- Finding: supervised training consistently beat self-supervised (MAE, HuBERT, SimCLR)

## Competition Performance (BirdCLEF+ 2025)
- DS@GT team: Perch TFLite → **0.729 public / 0.711 private ROC-AUC**
- For comparison: BirdSetEfficientNetB1 scored 0.810/0.778
- Perch alone scored ~60% accuracy — many target species (mammals, insects, amphibians) were outside its training set

## CPU Inference Performance
- **TFLite conversion gives ~10x speedup** over raw TF SavedModel
- DS@GT ran Perch TFLite in **~16 minutes** for ~700 min of audio
- Per-file: ~1.4 seconds
- Leaves ~74 minutes for downstream classifier within 90-min budget

## Integration Strategy

### Recommended Approach
1. **Pre-compute embeddings** for all training audio (5-sec chunks) using Perch v2 — one-time cost on GPU
2. **Train lightweight classifier** (logistic regression, 1-2 layer MLP, or XGBoost) on frozen 1536-d embeddings
3. **At inference**: Extract embeddings via TFLite (~16-20 min) → classify (~5-10 min)

### Hybrid Approach (recommended for BirdCLEF 2026)
- Use Perch logits directly for species within its ~15K label space
- Train custom SED model for species outside Perch's coverage (Pantanal-specific taxa)
- Ensemble both predictions

### Pre-computed Resources
- [BirdCLEF+ 2025 Perch Embeddings](https://www.kaggle.com/datasets/carlolepelaars/birdclef-2025-perch-embeddings) by Carlo Lepelaars

## Strengths
- Fast CPU inference (well within 90-min budget)
- High-quality embeddings, linearly separable
- Broad species coverage including non-avian taxa
- Proven baseline (0.71+ AUC)

## Risks
- Pantanal may have species outside Perch's label space → need fallback model
- TensorFlow dependency if rest of pipeline is PyTorch
- Not a top solution alone in 2025 — needs ensembling with custom models

## Dependencies
- `tensorflow>=2.20.0rc0`, `tensorflow-hub`
- TFLite for inference optimization

## Sources
- [Perch v2 paper (arXiv:2508.04665)](https://arxiv.org/abs/2508.04665)
- [Kaggle model page](https://www.kaggle.com/models/google/bird-vocalization-classifier)
- [DS@GT paper (arXiv:2507.08236)](https://arxiv.org/abs/2507.08236)
- [opensoundscape tutorial](https://opensoundscape.org/en/latest/tutorials/training_birdnet_and_perch.html)
