"""
Evaluation harness for Pantanal Sentinel — BirdCLEF+ 2026.

Computes macro-averaged ROC-AUC over species presence in 5-second intervals,
matching the Kaggle competition scorer exactly.

Usage:
    python eval/evaluate.py --predictions predictions.csv --ground-truth ground_truth.csv
    python eval/evaluate.py --predictions-dir eval/outputs/ --ground-truth ground_truth.csv

CSV format (both predictions and ground truth):
    row_id, species_1, species_2, ..., species_N
    Where row_id = "{filename}_{end_time}" (e.g., "soundscape_001_5")
    and each species column contains a probability [0, 1] (predictions)
    or binary label {0, 1} (ground truth).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray, species_names: list[str]) -> dict:
    """Compute macro-averaged ROC-AUC, matching Kaggle's sklearn-based scorer.

    For each species, computes ROC-AUC individually, then averages.
    Species with only one class present in y_true are excluded (undefined AUC).

    Args:
        y_true: Binary ground truth array, shape (n_samples, n_species).
        y_score: Predicted probabilities, shape (n_samples, n_species).
        species_names: List of species column names.

    Returns:
        Dictionary with 'macro_auc', 'per_species' dict, 'n_scored', 'n_skipped'.
    """
    from sklearn.metrics import roc_auc_score

    per_species = {}
    scored = []
    skipped = []

    for i, species in enumerate(species_names):
        true_col = y_true[:, i]
        score_col = y_score[:, i]

        # Skip species with only one class (AUC undefined)
        if len(np.unique(true_col)) < 2:
            skipped.append(species)
            continue

        auc = roc_auc_score(true_col, score_col)
        per_species[species] = auc
        scored.append(auc)

    macro_auc = float(np.mean(scored)) if scored else 0.0

    return {
        "macro_auc": macro_auc,
        "per_species": per_species,
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "skipped_species": skipped,
    }


def load_and_validate(pred_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load prediction and ground-truth CSVs, validate alignment."""
    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    # Validate row_id column exists
    if "row_id" not in pred_df.columns or "row_id" not in gt_df.columns:
        raise ValueError("Both CSVs must have a 'row_id' column")

    # Align on row_id
    pred_df = pred_df.set_index("row_id").sort_index()
    gt_df = gt_df.set_index("row_id").sort_index()

    # Check row alignment
    if not pred_df.index.equals(gt_df.index):
        missing_in_pred = set(gt_df.index) - set(pred_df.index)
        extra_in_pred = set(pred_df.index) - set(gt_df.index)
        msg = []
        if missing_in_pred:
            msg.append(f"Missing {len(missing_in_pred)} rows in predictions")
        if extra_in_pred:
            msg.append(f"Extra {len(extra_in_pred)} rows in predictions")
        raise ValueError("; ".join(msg))

    # Species columns = all columns except row_id (already removed by set_index)
    species_cols = sorted(gt_df.columns.tolist())

    # Ensure predictions have all species columns
    missing_species = set(species_cols) - set(pred_df.columns)
    if missing_species:
        raise ValueError(f"Predictions missing species columns: {missing_species}")

    # Align column order
    y_true = gt_df[species_cols].values.astype(np.float64)
    y_score = pred_df[species_cols].values.astype(np.float64)

    return y_true, y_score, species_cols


def evaluate_from_arrays(
    y_true: np.ndarray, y_score: np.ndarray, species_names: list[str], verbose: bool = True
) -> dict:
    """Run full evaluation and optionally print results."""
    results = macro_roc_auc(y_true, y_score, species_names)

    if verbose:
        print(f"Macro ROC-AUC: {results['macro_auc']:.6f}")
        print(f"Species scored: {results['n_scored']}, skipped: {results['n_skipped']}")

        if results["per_species"]:
            sorted_species = sorted(results["per_species"].items(), key=lambda x: x[1])
            print(f"\nWorst 10 species:")
            for name, auc in sorted_species[:10]:
                print(f"  {name}: {auc:.4f}")
            print(f"\nBest 10 species:")
            for name, auc in sorted_species[-10:]:
                print(f"  {name}: {auc:.4f}")

    return results


def evaluate_file(pred_path: str | Path, gt_path: str | Path, verbose: bool = True) -> dict:
    """Convenience function: load CSVs and evaluate."""
    y_true, y_score, species = load_and_validate(Path(pred_path), Path(gt_path))
    return evaluate_from_arrays(y_true, y_score, species, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 ROC-AUC Evaluator")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--output-tsv", type=str, help="Append result to TSV file")
    parser.add_argument("--experiment-name", type=str, default="", help="Name for TSV logging")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    results = evaluate_file(args.predictions, args.ground_truth, verbose=not args.quiet)

    # Optionally append to results TSV
    if args.output_tsv:
        tsv_path = Path(args.output_tsv)
        write_header = not tsv_path.exists()
        with open(tsv_path, "a") as f:
            if write_header:
                f.write("experiment\tmacro_auc\tn_scored\tn_skipped\n")
            f.write(
                f"{args.experiment_name}\t{results['macro_auc']:.6f}"
                f"\t{results['n_scored']}\t{results['n_skipped']}\n"
            )

    # Exit with non-zero if AUC is terrible (useful in autoresearch loop)
    sys.exit(0)


if __name__ == "__main__":
    main()
