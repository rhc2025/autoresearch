"""
Training loop for Pantanal Sentinel — BirdCLEF+ 2026.

This is the primary edit surface for the autoresearch training loop.
Modify hyperparameters, loss functions, augmentation, and architecture here.

Usage:
    python models/train.py --config models/configs/baseline.yaml
    python models/train.py --data-dir data/raw --epochs 30

The autoresearch loop:
    1. Modify this file (hyperparams, architecture, augmentation)
    2. git commit
    3. python models/train.py > run.log 2>&1
    4. Extract: grep "roc_auc" run.log
    5. If improved → keep commit; else → git reset
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ─── Hyperparameters (THE EDIT SURFACE) ───
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 30
BACKBONE = "tf_efficientnet_b0_ns"
DROPOUT = 0.3
MIXUP_ALPHA = 0.4  # 0 to disable mixup
LABEL_SMOOTHING = 0.05
SCHEDULER = "cosine"  # "cosine" or "onecycle"
WARMUP_EPOCHS = 3
SEED = 42
TIME_BUDGET = 1800  # 30 minutes max per experiment


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Apply mixup augmentation."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * y + (1 - lam) * y[idx]
    return x_mixed, y_mixed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_alpha: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mel, target in loader:
        mel, target = mel.to(device), target.to(device)

        # Mixup augmentation
        mel, target = mixup_data(mel, target, mixup_alpha)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    species_names: list[str],
    device: torch.device,
) -> dict:
    """Run validation, return macro ROC-AUC and per-species breakdown."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_preds = []
    all_targets = []

    for mel, target in loader:
        mel = mel.to(device)
        probs = model.predict_proba(mel)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(target.numpy())

    y_score = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Compute per-species AUC (skip species with only one class)
    per_species = {}
    scored = []
    for i, species in enumerate(species_names):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        auc = roc_auc_score(y_true[:, i], y_score[:, i])
        per_species[species] = auc
        scored.append(auc)

    macro_auc = float(np.mean(scored)) if scored else 0.0

    return {
        "macro_auc": macro_auc,
        "per_species": per_species,
        "n_scored": len(scored),
    }


def main():
    parser = argparse.ArgumentParser(description="Pantanal Sentinel Training")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Training data directory")
    parser.add_argument("--metadata", type=str, default="data/raw/train_metadata.csv")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints")
    parser.add_argument("--species-list", type=str, default="data/species_list.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(args.device)
    start_time = time.time()

    # Load species list
    species_path = Path(args.species_list)
    if not species_path.exists():
        print(f"ERROR: Species list not found at {species_path}")
        print("Run data preparation first to generate species_list.json")
        sys.exit(1)

    with open(species_path) as f:
        species_list = json.load(f)
    n_classes = len(species_list)
    print(f"Training with {n_classes} species")

    # Build model
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.backbones.efficientnet import EfficientNetSED

    model = EfficientNetSED(
        model_name=BACKBONE,
        n_classes=n_classes,
        pretrained=True,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {BACKBONE}, params: {param_count:,}")

    # Build dataset
    from data.dataset import BirdCLEFDataset

    train_dataset = BirdCLEFDataset(
        audio_dir=args.data_dir,
        metadata_csv=args.metadata,
        species_list=species_list,
    )
    print(f"Training samples: {len(train_dataset)}")

    # TODO: proper cross-validation split by recording site
    # For now, use 90/10 random split
    n_val = max(1, len(train_dataset) // 10)
    n_train = len(train_dataset) - n_val
    train_split, val_split = torch.utils.data.random_split(
        train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None,  # TODO: compute class weights from label distribution
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_loader)
        )

    # Warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS * len(train_loader)
    )

    # Training loop
    best_auc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            print(f"TIME BUDGET EXCEEDED ({elapsed:.0f}s > {TIME_BUDGET}s), stopping")
            break

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_alpha=MIXUP_ALPHA if epoch >= WARMUP_EPOCHS else 0.0,
        )

        # Step schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Validate
        val_results = validate(model, val_loader, species_list, device)
        macro_auc = val_results["macro_auc"]

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"roc_auc={macro_auc:.6f} | "
            f"scored={val_results['n_scored']} | "
            f"lr={lr:.2e} | "
            f"time={elapsed:.0f}s"
        )

        # Save best
        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_auc": macro_auc,
                    "config": {
                        "backbone": BACKBONE,
                        "n_classes": n_classes,
                        "dropout": DROPOUT,
                        "lr": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                        "mixup_alpha": MIXUP_ALPHA,
                    },
                },
                output_dir / "best_model.pt",
            )
            print(f"  -> New best ROC-AUC: {best_auc:.6f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Best ROC-AUC: {best_auc:.6f}")


if __name__ == "__main__":
    main()
