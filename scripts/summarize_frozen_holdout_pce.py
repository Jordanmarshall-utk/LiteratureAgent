from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "frozen_holdout_v1"
)
DEFAULT_FIGURE_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "publication_figures"
    / "main"
)

LABELS = {
    "01_original_database": "Original",
    "02_after_google_drive": "+ curated corpus",
    "03_after_google_drive_plus_expansion": "+ web expansion",
}
SPLITS = {
    "doi_grouped": {
        "label": "DOI-disjoint holdout",
        "prediction_suffix": "_frozen_predictions.csv",
    },
    "row_random": {
        "label": "Row-random holdout",
        "prediction_suffix": "_frozen_row_random_predictions.csv",
    },
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def load_predictions(
    input_dir: Path,
    split_strategy: str,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    suffix = SPLITS[split_strategy]["prediction_suffix"]
    for key in LABELS:
        path = input_dir / f"{key}{suffix}"
        frame = pd.read_csv(path)
        frame["group"] = frame["group"].astype(str)
        frame["row_key"] = frame["row_key"].astype(str)
        output[key] = frame
    baseline = output["01_original_database"]
    for key, frame in output.items():
        if len(frame) != len(baseline):
            raise RuntimeError(f"Prediction length mismatch for {key}")
        if not np.allclose(frame["actual"], baseline["actual"], equal_nan=True):
            raise RuntimeError(f"Actual-value mismatch for {key}")
        if not frame["group"].equals(baseline["group"]):
            raise RuntimeError(f"Frozen DOI-group mismatch for {key}")
        if not frame["row_key"].equals(baseline["row_key"]):
            raise RuntimeError(f"Frozen row-key mismatch for {key}")
    return output


def cluster_bootstrap(
    predictions: dict[str, pd.DataFrame],
    *,
    split_strategy: str,
    n_bootstrap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = predictions["01_original_database"]
    groups = baseline["group"].drop_duplicates().to_numpy()
    group_indices = {
        group: baseline.index[baseline["group"].eq(group)].to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(random_state)
    draws: list[dict[str, object]] = []

    for iteration in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        for key, frame in predictions.items():
            values = metrics(
                frame.loc[indices, "actual"].to_numpy(),
                frame.loc[indices, "predicted"].to_numpy(),
            )
            draws.append({"iteration": iteration, "dataset_version": key, **values})

    draws_frame = pd.DataFrame(draws)
    rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    baseline_point = metrics(
        baseline["actual"].to_numpy(), baseline["predicted"].to_numpy()
    )
    baseline_draws = draws_frame[
        draws_frame["dataset_version"].eq("01_original_database")
    ].set_index("iteration")

    for key, frame in predictions.items():
        point = metrics(frame["actual"].to_numpy(), frame["predicted"].to_numpy())
        subset = draws_frame[draws_frame["dataset_version"].eq(key)]
        for metric_name in ("r2", "rmse", "mae"):
            low, high = subset[metric_name].quantile([0.025, 0.975])
            rows.append(
                {
                    "dataset_version": key,
                    "split_strategy": split_strategy,
                    "label": LABELS[key],
                    "metric": metric_name,
                    "value": point[metric_name],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "frozen_test_rows": len(frame),
                    "frozen_test_doi_groups": len(groups),
                }
            )
            if key == "01_original_database":
                continue
            aligned = subset.set_index("iteration")
            delta = aligned[metric_name] - baseline_draws[metric_name]
            delta_low, delta_high = delta.quantile([0.025, 0.975])
            delta_rows.append(
                {
                    "dataset_version": key,
                    "split_strategy": split_strategy,
                    "label": LABELS[key],
                    "metric": metric_name,
                    "delta_vs_original": point[metric_name]
                    - baseline_point[metric_name],
                    "delta_ci_low": float(delta_low),
                    "delta_ci_high": float(delta_high),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def plot_metrics(summary: pd.DataFrame, output_stem: Path) -> None:
    order = list(LABELS)
    colors = ["#202124", "#777777", "#0F766E"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.5), dpi=180)
    settings = [
        ("r2", "R$^2$", "Higher is better"),
        ("rmse", "RMSE (PCE points)", "Lower is better"),
        ("mae", "MAE (PCE points)", "Lower is better"),
    ]
    for row_index, split_strategy in enumerate(SPLITS):
        split_summary = summary[summary["split_strategy"].eq(split_strategy)]
        for ax, (metric_name, ylabel, note) in zip(axes[row_index], settings):
            subset = split_summary[split_summary["metric"].eq(metric_name)].set_index(
                "dataset_version"
            ).loc[order]
            x = np.arange(len(order))
            values = subset["value"].to_numpy()
            low = values - subset["ci_low"].to_numpy()
            high = subset["ci_high"].to_numpy() - values
            ax.errorbar(
                x,
                values,
                yerr=np.vstack([low, high]),
                fmt="none",
                ecolor="#555555",
                capsize=4,
                linewidth=1.4,
                zorder=2,
            )
            ax.scatter(
                x,
                values,
                s=78,
                c=colors,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            ax.plot(x, values, color="#999999", linewidth=1.0, zorder=1)
            for xpos, value in zip(x, values):
                ax.annotate(
                    f"{value:.3f}",
                    (xpos, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )
            ax.set_xticks(x, [LABELS[key] for key in order], rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(note, fontsize=10, color="#555555")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axes[row_index, 0].text(
            -0.28,
            0.5,
            SPLITS[split_strategy]["label"],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        "PCE performance on fixed, common holdouts",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Each row uses one baseline-derived test population across all three training stages; bars show 95% DOI-cluster bootstrap intervals.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def write_summary(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Frozen PCE holdout comparisons",
        "",
        "Both test populations are selected once from the original model-ready database and remain unchanged across the original, curated, and web-expanded training stages.",
    ]
    for split_strategy in SPLITS:
        split_summary = summary[summary["split_strategy"].eq(split_strategy)]
        values = split_summary.pivot(
            index="dataset_version", columns="metric", values="value"
        )
        test_rows = int(split_summary["frozen_test_rows"].iloc[0])
        test_groups = int(split_summary["frozen_test_doi_groups"].iloc[0])
        lines.extend(
            [
                "",
                f"## {SPLITS[split_strategy]['label']}",
                "",
                f"Fixed test population: {test_rows:,} rows from {test_groups:,} DOI groups.",
                "",
                "| Dataset | R2 | RMSE | MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        for key in LABELS:
            row = values.loc[key]
            lines.append(
                f"| {LABELS[key]} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: database growth increased training support but did not improve performance on either frozen original-database holdout.",
            "The DOI-disjoint split estimates transfer to unseen publications, whereas the row-random split estimates interpolation when other rows from the same publications may remain in training.",
            "This analysis isolates the training-data change; it should not be interpreted as a universal ranking of corpus quality.",
            "",
            "Paired DOI-cluster bootstrap deltas are retained in `frozen_holdout_paired_deltas.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    summaries = []
    delta_frames = []
    split_audit = {}
    for split_strategy in SPLITS:
        predictions = load_predictions(args.input_dir, split_strategy)
        summary, deltas = cluster_bootstrap(
            predictions,
            split_strategy=split_strategy,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
        summaries.append(summary)
        delta_frames.append(deltas)
        baseline = predictions["01_original_database"]
        split_audit[split_strategy] = {
            "test_rows": int(len(baseline)),
            "test_doi_groups": int(baseline["group"].nunique()),
        }
    summary = pd.concat(summaries, ignore_index=True)
    deltas = pd.concat(delta_frames, ignore_index=True)
    summary.to_csv(args.input_dir / "frozen_holdout_metrics_with_ci.csv", index=False)
    deltas.to_csv(args.input_dir / "frozen_holdout_paired_deltas.csv", index=False)
    write_summary(summary, deltas, args.input_dir / "FROZEN_HOLDOUT_SUMMARY.md")
    stem = args.figure_dir / "figure_pce_frozen_holdout"
    plot_metrics(summary, stem)
    audit = {
        "n_bootstrap": args.n_bootstrap,
        "random_state": args.random_state,
        "prediction_sets": list(LABELS),
        "split_strategies": split_audit,
        "figure": str(stem.with_suffix(".png")),
    }
    (args.input_dir / "frozen_holdout_bootstrap_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
