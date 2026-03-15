# BirdCLEF+ 2025 — Winning Solution Teardowns

## Leaderboard Summary

| Place | Team | Private AUC | Key Innovation |
|-------|------|-------------|----------------|
| 1st | Nikita Babych (solo) | ~0.933 | Multi-iterative Noisy Student + taxonomic pipelines |
| 2nd | Sydorskyi & Goncalves | 0.928 | Domain shift handling + semi-supervised distillation |
| 5th | myso1987 | — | 3-stage SED pipeline + OpenVINO |
| 38th | Max Melichov (top 2%) | 0.902 | Quantile-Mix blending + TTA |

---

## 1st Place — Nikita Babych (~0.933)

**Title:** "Multi-Iterative Noisy Student Is All You Need"

### Architecture
- Ensemble of EfficientNet-L0, B4, B3; RegNetY-016, RegNetY-008
- Separate EfficientNet-B0 pipeline for amphibians/insects (taxonomic-specific)
- SED heads for frame-level annotation refinement

### Training Strategy
- Baseline: CE loss, AdamW, Cosine scheduling, EfficientNet-B0 + RegNetY-008 → 0.872 AUC
- Pseudo-labeling Round 1 + MixUp + StochasticDepth → 0.898
- Multi-Iterative Noisy Student (4 rounds + Power Scaling) → 0.930
- Taxonomic-specific pipeline for insects/amphibians → +0.003 → 0.933
- PowerTransform on pseudo-labels enables progressive refinement across rounds
- Validated on public LB (no reliable local validation found)

### Key Takeaway
Single biggest lever was **4 rounds of iterative pseudo-labeling with power scaling** (+0.058 AUC). Taxonomic splitting added marginal but consistent gain.

### Source
- [Kaggle writeup](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)

---

## 2nd Place — Sydorskyi & Goncalves (0.928)

**Title:** "Tackling Domain Shift via Transfer Learning and Semi-Supervised Distillation"

### Architecture
- `eca_nfnet_l0` (Normalizer-Free Network with ECA attention)
- `tf_efficientnetv2_s_in21k` (ImageNet-21k pretrained)

### Training Strategy
- Focal BCE Loss
- Optimizers: Radan (for NFNet), AdamW at 1e-4 (for EfficientNetV2)
- Cosine batch-level LR scheduling (floor 1e-6)
- Batch size 64, 5-second clips, 50 epochs
- Two balancing strategies: sqrt-based and equal balancing
- Pre-training on extended datasets from Xeno-Canto, iNaturalist, CSA
- "AddRareBirdsNoLeak" — rare species augmentation without data leakage
- Pseudo-labeling with confidence thresholds, iterated 2-3 times
- Audio preprocessed to HDF5 format

### Inference
- 5-fold cross-validation ensemble
- PyTorch → ONNX → FP16 OpenVINO conversion
- Published ready-to-infer OpenVINO models on Kaggle

### Key Takeaway
Careful **external data curation** and **domain shift mitigation** through pre-training. FP16 OpenVINO for CPU budget.

### Source
- [GitHub](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)

---

## 5th Place — myso1987

### Architecture
- Ensemble of EfficientNet-B0, B3, EfficientNetV2-B3, EfficientNetV2-S
- All trained as SED models

### Training Strategy
- Three-stage pipeline:
  1. Initial model training on labeled data
  2. Training with pseudo-labels (TSS variant)
  3. Further refinement with pseudo-labels
- Audio cropped into 30s and 60s clips
- Species with < 20 samples receive oversampling augmentation

### Inference
- Models converted to OpenVINO format

### Source
- [GitHub](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution)

---

## Common Patterns Across Winners

### Backbones (by frequency)
1. **EfficientNet family** (B0, B3, B4) — universal baseline, fast inference
2. **EfficientNetV2** (B3, S) — improved training efficiency, ImageNet-21k pretraining
3. **eca_nfnet_l0** — normalizer-free with ECA channel attention
4. **RegNetY** (008, 016) — good accuracy/speed tradeoff
5. **MNASNet-100, SPNASNet-100** — NAS-derived lightweight models for diversity

### Augmentation
- **MixUp** (additive, max-of-labels strategy) — universal
- **Time shifting** (1-second window)
- **Random cropping** (5-second clips from longer recordings)
- **StochasticDepth** (drop-path regularization)
- Minimal exotic augmentation — top teams kept it simple

### Loss Functions
- **Focal BCE** — most common among top teams
- **CE** — BirdCLEF 2024 1st found CE vastly superior when most samples are single-class
- **SoftAUCLoss** — directly optimizes ROC-AUC; resistant to overfitting; supports soft labels
- Key trick: **Train with softmax (CE), infer with sigmoid** (2024 1st place)

### CPU Inference (90-min Budget)
1. PyTorch → ONNX → OpenVINO FP16 (2nd, 5th place)
2. Small backbones: EfficientNet-B0/EfficientViT-b0 can do 5-fold in ~40 min via ONNX
3. Ensemble size: typically 2-4 models (not 10+)
4. FP16 preferred over INT8 (preserves accuracy, INT8 not widely used by winners)
5. TFLite as alternative (DS@GT: 10x speedup for Perch)

### Handling Rare Species
1. Oversampling species with < 20-30 samples
2. External data from Xeno-Canto, iNaturalist, previous years (capped 500/species)
3. Sqrt/equal class balancing in sampling
4. Focal BCE Loss (down-weight easy, focus on hard/rare)
5. Silero VAD for data cleaning (remove human voice fragments)
6. Taxonomic-specific pipelines (1st place)

---

## Implications for Pantanal Sentinel (BirdCLEF+ 2026)

### Must-Have Techniques
1. **Iterative pseudo-labeling** (4+ rounds) — the biggest single lever
2. **Focal BCE or SoftAUC loss** — standard for this competition
3. **EfficientNet-B0/B3 backbone** — proven, fast, well-understood
4. **OpenVINO FP16 export** — fits 2-4 model ensemble in 90 min
5. **External data** from Xeno-Canto + previous BirdCLEF years

### Should-Have Techniques
1. **Taxonomic-specific pipelines** (birds vs. insects/amphibians)
2. **Perch v2 embeddings** as supplementary signal
3. **SED heads** for frame-level refinement
4. **TTA** with time shifts (+0.012 AUC observed)

### Nice-to-Have
1. **Ecological priors** (geographic/temporal filtering) — not used by 2025 winners but could help
2. **BioME foundation model** — if available for Pantanal taxa
3. **NAS-derived lightweight models** for ensemble diversity

---

## References

- [1st Place (Kaggle)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [2nd Place (GitHub)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [5th Place (GitHub)](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution)
- [Top-5 Overview (Tekkix)](https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a)
- [Top 2% Writeup (Medium)](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
- [DS@GT Paper (arXiv)](https://arxiv.org/abs/2507.08236)
- [BirdCLEF 2024 3rd Place (GitHub)](https://github.com/TheoViel/kaggle_birdclef2024)
- [BirdCLEF+ 2025 Overview (ResearchGate)](https://www.researchgate.net/publication/396180283_Overview_of_BirdCLEF_2025_Multi-Taxonomic_Sound_Identification_in_the_Middle_Magdalena_Colombia)
