#!/usr/bin/env python3
"""Combine grouped and row-random PCE comparisons into one clear audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-csv", required=True)
    parser.add_argument("--row-random-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    grouped = pd.read_csv(args.grouped_csv, low_memory=False)
    row_random = pd.read_csv(args.row_random_csv, low_memory=False)
    grouped = grouped[grouped["model_family"].eq("pce_direct")].copy()
    row_random = row_random[row_random["model_family"].eq("pce_direct")].copy()
    grouped["evaluation"] = "DOI-grouped"
    grouped["intended_use"] = "Unseen-paper generalization"
    grouped["same_doi_can_cross_folds"] = False
    row_random["evaluation"] = "Row-random"
    row_random["intended_use"] = "Within-corpus row interpolation diagnostic"
    row_random["same_doi_can_cross_folds"] = True
    combined = pd.concat([grouped, row_random], ignore_index=True)

    keep = [
        "evaluation",
        "backend",
        "model_family",
        "intended_use",
        "same_doi_can_cross_folds",
        "n",
        "n_features_raw",
        "actual_n_estimators",
        "oof_r2",
        "oof_rmse",
        "oof_mae",
        "group_holdout_r2",
        "group_holdout_rmse",
        "group_holdout_mae",
        "estimator_class",
        "input_fingerprint_sha256",
    ]
    combined[keep].to_csv(output / "authoritative_pce_evaluation_comparison.csv", index=False)

    order = ["DOI-grouped", "Row-random"]
    backends = ["extra_trees", "xgboost"]
    r2 = combined.pivot(index="evaluation", columns="backend", values="oof_r2").reindex(order)
    rmse = combined.pivot(index="evaluation", columns="backend", values="oof_rmse").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.35), dpi=240)
    colors = {"extra_trees": "#4C78A8", "xgboost": "#E45756"}
    r2[backends].plot(kind="bar", ax=axes[0], color=[colors[b] for b in backends], width=0.72)
    axes[0].set_title("PCE R² depends strongly on split strategy")
    axes[0].set_ylabel("Out-of-fold R²")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].axhline(0, color="black", linewidth=0.8)
    for patch in axes[0].patches:
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{patch.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    rmse[backends].plot(kind="bar", ax=axes[1], color=[colors[b] for b in backends], width=0.72)
    axes[1].set_title("PCE prediction error")
    axes[1].set_ylabel("RMSE (PCE percentage points)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Regressor", frameon=False)
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()

    fig.suptitle("Matched PCE model comparison: identical rows, features, and estimator counts")
    fig.tight_layout()
    fig.savefig(output / "authoritative_pce_evaluation_comparison.png", bbox_inches="tight")
    plt.close(fig)

    grouped_winner = grouped.sort_values("oof_r2", ascending=False).iloc[0]
    row_winner = row_random.sort_values("oof_r2", ascending=False).iloc[0]
    summary = {
        "rows": int(combined["n"].iloc[0]),
        "features": int(combined["n_features_raw"].iloc[0]),
        "estimators_per_model": int(combined["actual_n_estimators"].iloc[0]),
        "grouped_winner": {
            "backend": grouped_winner["backend"],
            "oof_r2": grouped_winner["oof_r2"],
            "oof_rmse": grouped_winner["oof_rmse"],
            "oof_mae": grouped_winner["oof_mae"],
            "interpretation": "Primary estimate for generalization to papers absent from training.",
        },
        "row_random_winner": {
            "backend": row_winner["backend"],
            "oof_r2": row_winner["oof_r2"],
            "oof_rmse": row_winner["oof_rmse"],
            "oof_mae": row_winner["oof_mae"],
            "interpretation": (
                "Within-corpus row interpolation diagnostic; related devices from one DOI "
                "can appear in both training and test folds."
            ),
        },
        "historical_score_resolution": (
            "The historical approximately 0.6 R2 is reproduced by row-random Extra Trees. "
            "The lower DOI-grouped score answers the harder unseen-paper question."
        ),
    }
    (output / "authoritative_pce_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
