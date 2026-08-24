#!/usr/bin/env python3
"""Summarize matched original-versus-integrated PCE model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RUNS = (
    ("Original", "DOI-grouped", "original_grouped"),
    ("Integrated", "DOI-grouped", "integrated_grouped"),
    ("Original", "Row-random", "original_row_random"),
    ("Integrated", "Row-random", "integrated_row_random"),
)

MODELS = (
    ("Direct", Path("pce/pce_direct_metrics.json")),
    (
        "Hierarchical residual",
        Path("pce/hierarchical_residual/pce_hierarchical_residual_metrics.json"),
    ),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for dataset, evaluation, folder in RUNS:
        for model, relative_path in MODELS:
            metrics = load_json(args.run_root / folder / relative_path)
            rows.append(
                {
                    "dataset": dataset,
                    "evaluation": evaluation,
                    "model": model,
                    "model_rows": int(metrics["n"]),
                    "oof_r2": float(metrics["oof_r2"]),
                    "oof_rmse": float(metrics["oof_rmse"]),
                    "oof_mae": float(metrics["oof_mae"]),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.run_root / "matched_pce_before_after_metrics.csv", index=False)

    deltas = []
    for (evaluation, model), group in metrics.groupby(["evaluation", "model"]):
        before = group.set_index("dataset").loc["Original"]
        after = group.set_index("dataset").loc["Integrated"]
        deltas.append(
            {
                "evaluation": evaluation,
                "model": model,
                "rows_added": int(after.model_rows - before.model_rows),
                "delta_r2": float(after.oof_r2 - before.oof_r2),
                "delta_rmse": float(after.oof_rmse - before.oof_rmse),
                "delta_mae": float(after.oof_mae - before.oof_mae),
            }
        )
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(args.run_root / "matched_pce_before_after_deltas.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    colors = {"Original": "#4C78A8", "Integrated": "#E45756"}
    for ax, evaluation in zip(axes, ("DOI-grouped", "Row-random")):
        subset = metrics[metrics.evaluation.eq(evaluation)]
        x = range(2)
        width = 0.34
        for offset, dataset in ((-width / 2, "Original"), (width / 2, "Integrated")):
            values = [
                subset[(subset.dataset.eq(dataset)) & (subset.model.eq(model))].oof_r2.iloc[0]
                for model, _ in MODELS
            ]
            bars = ax.bar([i + offset for i in x], values, width, label=dataset, color=colors[dataset])
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(list(x), ["Direct", "Hierarchical\nresidual"])
        ax.set_ylabel("Out-of-fold R²")
        ax.set_title(evaluation)
        ax.set_ylim(0, 0.72)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(args.run_root / "matched_pce_before_after_r2.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.run_root / "matched_pce_before_after_r2.svg", bbox_inches="tight")
    plt.close(fig)

    direct = metrics[metrics.model.eq("Direct")]
    lines = [
        "# Matched PCE Before/After Comparison",
        "",
        "All runs use the 2018+ filter, Extra Trees, 300 estimators, and the same feature pipeline.",
        "DOI-grouped CV estimates generalization to unseen papers; row-random CV measures within-corpus interpolation.",
        "",
        markdown_table(metrics),
        "",
        "## Deltas",
        "",
        markdown_table(delta_df),
        "",
        f"The integrated direct model used {int(direct.model_rows.max() - direct.model_rows.min())} additional PCE rows.",
        "The matched accuracy differences are small and inconsistent, so this batch increased target support but did not show a consistent aggregate accuracy improvement.",
    ]
    (args.run_root / "MATCHED_PCE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
