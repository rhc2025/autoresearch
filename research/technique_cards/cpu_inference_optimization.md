# Technique Card: CPU Inference Optimization (ONNX/OpenVINO FP16)

## Summary
Export trained PyTorch models to ONNX, then convert to OpenVINO FP16 format for fast CPU inference. This pipeline enables fitting 2-4 model ensembles within the 90-minute Kaggle CPU budget.

## Source
- 2nd Place BirdCLEF+ 2025 (Sydorskyi & Goncalves)
- 5th Place BirdCLEF+ 2025 (myso1987)
- 3rd Place BirdCLEF 2024 (Henkel/Puget/Viel)

## Performance Benchmarks

| Format | Model | 5-fold time | Notes |
|--------|-------|-------------|-------|
| ONNX FP32 | EfficientViT-b0 | ~40 min | BirdCLEF 2024 3rd place |
| OpenVINO FP16 | eca_nfnet_l0 | ~20 min | BirdCLEF+ 2025 2nd place |
| TFLite | Perch v2 | ~16 min | DS@GT (10x vs raw TF) |
| ONNX FP32 | EfficientNet-B0 | ~25 min | Estimated |

## Pipeline

### Step 1: PyTorch → ONNX
```python
import torch

dummy_input = torch.randn(1, 1, 128, 313)  # (batch, channels, n_mels, time)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

### Step 2: ONNX → OpenVINO FP16
```python
from openvino.tools import mo

ov_model = mo.convert_model("model.onnx", compress_to_fp16=True)
from openvino.runtime import serialize
serialize(ov_model, "model_fp16.xml")
```

### Step 3: Inference
```python
from openvino.runtime import Core

core = Core()
model = core.read_model("model_fp16.xml")
compiled = core.compile_model(model, "CPU")
infer_request = compiled.create_infer_request()

# Batch inference
result = infer_request.infer({0: mel_batch})
```

## Key Decisions

### ONNX vs OpenVINO
- OpenVINO FP16 is ~1.5-2x faster than ONNX FP32 on Intel CPUs
- OpenVINO may lose ~0.01 AUC vs ONNX FP32 (BirdCLEF 2024 3rd place observed this)
- For ensemble submissions, the speed gain usually outweighs the accuracy loss

### FP16 vs INT8
- **FP16 preferred** by all top BirdCLEF+ 2025 teams
- INT8 requires calibration dataset and may lose more accuracy
- FP16 sufficient to fit 2-4 models in 90-min budget
- Consider INT8 only if budget is extremely tight

### Backbone Selection for CPU
Best throughput/accuracy tradeoffs:
1. EfficientNet-B0 — fastest, solid baseline
2. EfficientViT-b0 — competitive speed, good for diversity
3. MNASNet-100 — very fast, NAS-optimized
4. eca_nfnet_l0 — slightly slower but strong accuracy

### Ensemble Budget Planning
```
Total budget: 90 minutes
Audio loading + preprocessing: ~10 min
Model inference (4 models × 5 folds): ~60 min  → 3 min per model-fold
Post-processing + submission write: ~5 min
Safety margin: ~15 min
```

## Risks
- Kaggle CPU specs may differ from local CPU — always profile on Kaggle
- OpenVINO version compatibility (pin version in submission notebook)
- Dynamic batching may not work well with OpenVINO — use fixed batch sizes
- Some PyTorch ops may not export cleanly to ONNX (attention layers)

## Dependencies
- `onnx`, `onnxruntime` (for ONNX path)
- `openvino`, `openvino-dev` (for OpenVINO path)
- `torch` (for export)

## Priority for Pantanal Sentinel
**HIGH** — Start basic ONNX export tests in Sprint 2. Full optimization in Sprint 4. Must be validated on Kaggle before final submission.
