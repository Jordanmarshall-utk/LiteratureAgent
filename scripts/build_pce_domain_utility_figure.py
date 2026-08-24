from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["text.color"] = "#000000"
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "frozen_holdout_harmonized_v2"
)
EXPANDED_DIR = HISTORICAL_DIR.parent / "harmonized_expansion_v1"
PUBLICATION_MAIN = (
    ROOT / "artifacts" / "literature_agent_pce_stability" / "publication_figures" / "main"
)
PUBLICATION_SUPP = PUBLICATION_MAIN.parent / "supplementary"
DELIVERABLE = ROOT / "deliverables" / "LiteratureAgent" / "figures"
PANEL_DELIVERABLE = DELIVERABLE / "figure_4_panels"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, iterations: int = 2000, random_state: int = 42
) -> list[dict[str, float | str]]:
    groups = frame["group"].astype("string").fillna("missing_group")
    unique_groups = groups.unique()
    group_indices = {
        group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups
    }
    rng = np.random.default_rng(random_state)
    draws = {"r2": [], "rmse": [], "mae": []}
    actual_all = frame["actual"].to_numpy(dtype=float)
    predicted_all = frame["predicted"].to_numpy(dtype=float)
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        values = metrics(actual_all[indices], predicted_all[indices])
        for metric, value in values.items():
            draws[metric].append(value)
    point = metrics(actual_all, predicted_all)
    rows = []
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "value": point[metric],
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
            }
        )
    return rows


def add_metric_panel(
    axis,
    table: pd.DataFrame,
    labels: list[str],
    colors: list[str],
    metric: str,
    title: str,
) -> None:
    subset = table.loc[table["metric"].eq(metric)].copy()
    x = np.arange(len(subset))
    values = subset["value"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            values - subset["ci_low"].to_numpy(dtype=float),
            subset["ci_high"].to_numpy(dtype=float) - values,
        ]
    )
    axis.bar(x, values, color=colors, edgecolor="#202124", linewidth=0.8, width=0.68)
    axis.errorbar(
        x,
        values,
        yerr=errors,
        fmt="none",
        ecolor="#202124",
        elinewidth=1.1,
        capsize=3,
    )
    if metric == "r2":
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.set_ylabel("R²")
    elif metric == "rmse":
        axis.set_ylabel("RMSE (PCE points)")
    else:
        axis.set_ylabel("MAE (PCE points)")
    axis.set_xticks(x, labels)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=9)
    for index, value in enumerate(values):
        offset = 0.025 * max(1.0, float(np.ptp(values)) + abs(value))
        va = "bottom" if value >= 0 else "top"
        axis.text(index, value + (offset if value >= 0 else -offset), f"{value:.3f}", ha="center", va=va, fontsize=8)


def save_standalone_panel(
    table: pd.DataFrame,
    labels: list[str],
    colors: list[str],
    metric: str,
    title: str,
    note: str,
    publication_stem: str,
    deliverable_stem: str,
) -> None:
    fig, axis = plt.subplots(figsize=(5.6, 4.6))
    add_metric_panel(axis, table, labels, colors, metric, title)
    fig.text(0.5, 0.018, note, ha="center", fontsize=8.5, color="#000000")
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    publication_png = PUBLICATION_MAIN / f"{publication_stem}.png"
    publication_pdf = PUBLICATION_MAIN / f"{publication_stem}.pdf"
    fig.savefig(publication_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(publication_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(publication_png, PANEL_DELIVERABLE / f"{deliverable_stem}.png")
    shutil.copy2(publication_pdf, PANEL_DELIVERABLE / f"{deliverable_stem}.pdf")


def main() -> None:
    PUBLICATION_MAIN.mkdir(parents=True, exist_ok=True)
    PUBLICATION_SUPP.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    PANEL_DELIVERABLE.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(HISTORICAL_DIR / "frozen_holdout_metrics_with_ci.csv")
    historical = historical.loc[historical["split_strategy"].eq("doi_grouped")].copy()
    historical["domain"] = "historical"
    historical["model"] = historical["dataset_version"].map(
        {
            "01_original_database": "Original",
            "02_after_google_drive": "Curated expansion",
            "03_after_google_drive_plus_expansion": "Full expansion",
        }
    )
    historical_order = [
        "01_original_database",
        "02_after_google_drive",
        "03_after_google_drive_plus_expansion",
    ]
    historical["order"] = historical["dataset_version"].map(
        {value: index for index, value in enumerate(historical_order)}
    )
    historical = historical.sort_values(["metric", "order"])

    expanded_rows = []
    expanded_files = {
        "Original-only training": EXPANDED_DIR / "expanded_domain_original_only_predictions.csv",
        "Original + accepted expansion": EXPANDED_DIR
        / "expanded_domain_original_plus_accepted_expansion_predictions.csv",
    }
    for order, (label, path) in enumerate(expanded_files.items()):
        for row in cluster_bootstrap(pd.read_csv(path)):
            expanded_rows.append(
                {
                    "domain": "accepted_literatureagent",
                    "model": label,
                    "order": order,
                    **row,
                }
            )
    expanded = pd.DataFrame(expanded_rows).sort_values(["metric", "order"])

    source_data = pd.concat(
        [
            historical[["domain", "model", "metric", "value", "ci_low", "ci_high"]],
            expanded[["domain", "model", "metric", "value", "ci_low", "ci_high"]],
        ],
        ignore_index=True,
    )
    source_data.to_csv(PUBLICATION_MAIN / "figure_pce_domain_utility_source_data.csv", index=False)

    historical_colors = ["#4C78A8", "#72B7B2", "#E69F00"]
    expanded_colors = ["#7A7A7A", "#2A9D8F"]
    metric_names = {"r2": "R²", "rmse": "RMSE", "mae": "MAE"}
    for panel, metric in zip(["A", "B", "C"], ["r2", "rmse", "mae"]):
        save_standalone_panel(
            historical.loc[historical["metric"].eq(metric)],
            ["Original", "Curated", "Full"],
            historical_colors,
            metric,
            f"{panel}  Historical-domain {metric_names[metric]}",
            "Frozen holdout: 4,092 rows from 744 DOI groups; 95% DOI-cluster bootstrap CI.",
            f"figure_pce_historical_{metric}",
            f"figure_4{panel}_historical_{metric}",
        )
    for panel, metric in zip(["D", "E", "F"], ["r2", "rmse", "mae"]):
        save_standalone_panel(
            expanded.loc[expanded["metric"].eq(metric)],
            ["Original\nonly", "+ accepted\nexpansion"],
            expanded_colors,
            metric,
            f"{panel}  LiteratureAgent-domain {metric_names[metric]}",
            "Holdout: 125 accepted records from 80 unseen DOI groups; 95% DOI-cluster bootstrap CI.",
            f"figure_pce_literatureagent_{metric}",
            f"figure_4{panel}_literatureagent_{metric}",
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), constrained_layout=False)
    plt.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.12, wspace=0.34, hspace=0.48)
    for column, (metric, name) in enumerate(
        [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE")]
    ):
        add_metric_panel(
            axes[0, column],
            historical.loc[historical["metric"].eq(metric)],
            ["Original", "Curated", "Full"],
            historical_colors,
            metric,
            f"{chr(65 + column)}  Historical-domain {name}",
        )
        add_metric_panel(
            axes[1, column],
            expanded.loc[expanded["metric"].eq(metric)],
            ["Original\nonly", "+ accepted\nexpansion"],
            expanded_colors,
            metric,
            f"{chr(68 + column)}  LiteratureAgent-domain {name}",
        )
    fig.suptitle(
        "Schema-harmonized literature expansion preserves historical-domain performance\nand improves generalization to held-out LiteratureAgent records",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.505,
        "Frozen historical holdout: 4,092 rows from 744 DOI groups",
        ha="center",
        fontsize=10,
        color="#000000",
    )
    fig.text(
        0.5,
        0.055,
        "LiteratureAgent holdout: 125 accepted records from 80 unseen DOI groups. Error bars: 95% DOI-cluster bootstrap intervals.",
        ha="center",
        fontsize=9.5,
        color="#000000",
    )
    png = PUBLICATION_MAIN / "figure_pce_domain_utility.png"
    pdf = PUBLICATION_MAIN / "figure_pce_domain_utility.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    shutil.copy2(png, DELIVERABLE / "figure_4.png")
    shutil.copy2(pdf, DELIVERABLE / "figure_4.pdf")
    audit_png = EXPANDED_DIR / "figure_pce_harmonized_domain_evaluation.png"
    audit_pdf = EXPANDED_DIR / "figure_pce_harmonized_domain_evaluation.pdf"
    if audit_png.exists():
        shutil.copy2(audit_png, PUBLICATION_SUPP / "figure_s_pce_integration_audit.png")
    if audit_pdf.exists():
        shutil.copy2(audit_pdf, PUBLICATION_SUPP / "figure_s_pce_integration_audit.pdf")

    manifest_path = DELIVERABLE / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figure_4"] = {
        "png": str((DELIVERABLE / "figure_4.png").resolve()),
        "pdf": str((DELIVERABLE / "figure_4.pdf").resolve()),
        "analysis_source": str(png.resolve()),
        "source_data": str(
            (PUBLICATION_MAIN / "figure_pce_domain_utility_source_data.csv").resolve()
        ),
        "standalone_panels": {
            panel: str((PANEL_DELIVERABLE / filename).resolve())
            for panel, filename in {
                "4A": "figure_4A_historical_r2.png",
                "4B": "figure_4B_historical_rmse.png",
                "4C": "figure_4C_historical_mae.png",
                "4D": "figure_4D_literatureagent_r2.png",
                "4E": "figure_4E_literatureagent_rmse.png",
                "4F": "figure_4F_literatureagent_mae.png",
            }.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(source_data.to_string(index=False))


if __name__ == "__main__":
    main()
