#!/usr/bin/env python3
"""Compare matched PCE/stability model metrics before and after dataset updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HIGHER_IS_BETTER = [
    "oof_r2",
    "group_holdout_r2",
    "oof_accuracy",
    "oof_balanced_accuracy",
    "oof_f1",
    "oof_roc_auc",
]
LOWER_IS_BETTER = [
    "oof_rmse",
    "oof_mae",
    "group_holdout_rmse",
    "group_holdout_mae",
]
ALL_METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER
KEY_COLS = ["comparison_family", "mode", "target", "task"]
IGNORED_CONFIG_KEYS = {"INPUT_CSV", "OUTPUT_DIR"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_family(row: pd.Series) -> str:
    family = str(row.get("model_family", "")).strip()
    mode = str(row.get("mode", "")).strip()
    if family == "PCE" or str(row.get("target", "")).startswith("JV_"):
        return "PCE"
    if family in {"stability", "empirical_baseline", ""} and mode in {
        "design_only",
        "design_plus_predicted_pce",
        "design_plus_measured_initial_performance",
    }:
        return "empirical_baseline"
    return family or "unknown"


def load_run(root: Path, label: str) -> Tuple[pd.DataFrame, Dict, Dict]:
    metric_path = root / "model_comparison_pce_then_stability.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing model comparison CSV: {metric_path}")
    df = pd.read_csv(metric_path)
    for col in ["model_family", "mode", "target", "task", "status"]:
        if col not in df.columns:
            df[col] = ""
    df["comparison_family"] = df.apply(canonical_family, axis=1)
    df["run_label"] = label
    for metric in ALL_METRICS + ["n"]:
        if metric not in df.columns:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df, read_json(root / "config.json"), read_json(root / "run_environment.json")


def compare_settings(before_cfg: Dict, after_cfg: Dict, before_env: Dict, after_env: Dict) -> pd.DataFrame:
    rows = []
    for source, before, after, ignored in [
        ("config", before_cfg, after_cfg, IGNORED_CONFIG_KEYS),
        ("environment", before_env, after_env, {"input_csv"}),
    ]:
        for key in sorted((set(before) | set(after)) - ignored):
            b, a = before.get(key), after.get(key)
            rows.append({
                "source": source,
                "setting": key,
                "before": json.dumps(b, sort_keys=True),
                "after": json.dumps(a, sort_keys=True),
                "matches": b == a,
            })
    return pd.DataFrame(rows)


def make_aligned_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    keep = KEY_COLS + ["status", "n"] + ALL_METRICS
    b = before[keep].drop_duplicates(KEY_COLS).rename(columns={c: f"before_{c}" for c in keep if c not in KEY_COLS})
    a = after[keep].drop_duplicates(KEY_COLS).rename(columns={c: f"after_{c}" for c in keep if c not in KEY_COLS})
    aligned = b.merge(a, on=KEY_COLS, how="outer", indicator=True)
    aligned["matched_trained"] = (
        aligned["_merge"].eq("both")
        & aligned["before_status"].eq("trained")
        & aligned["after_status"].eq("trained")
    )
    aligned["n_delta"] = aligned["after_n"] - aligned["before_n"]
    return aligned


def make_metric_long(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in aligned.iterrows():
        for metric in ALL_METRICS:
            before = row.get(f"before_{metric}", np.nan)
            after = row.get(f"after_{metric}", np.nan)
            if not (np.isfinite(before) and np.isfinite(after)):
                continue
            higher = metric in HIGHER_IS_BETTER
            delta = float(after - before)
            rows.append({
                **{k: row[k] for k in KEY_COLS},
                "metric": metric,
                "before": float(before),
                "after": float(after),
                "raw_delta_after_minus_before": delta,
                "signed_improvement": delta if higher else -delta,
                "direction": "higher_is_better" if higher else "lower_is_better",
                "before_n": row.get("before_n"),
                "after_n": row.get("after_n"),
                "n_delta": row.get("n_delta"),
                "matched_trained": bool(row.get("matched_trained", False)),
            })
    return pd.DataFrame(rows)


def short_label(row: pd.Series) -> str:
    target = str(row["target"]).replace("target_", "").replace("_hours", "h")
    mode = str(row["mode"]).replace("design_plus_", "+").replace("design_", "")
    return f"{mode} | {target}"


def paired_bar_plot(df: pd.DataFrame, metric: str, title: str, path: Path, limit: int = 18) -> None:
    data = df[(df["metric"] == metric) & df["matched_trained"]].copy()
    if data.empty:
        return
    data["label"] = data.apply(short_label, axis=1)
    data = data.sort_values("signed_improvement", ascending=False).head(limit)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(data) + 1.8)), dpi=180)
    ax.barh(y - 0.18, data["before"], height=0.34, label="Before literature update")
    ax.barh(y + 0.18, data["after"], height=0.34, label="After literature update")
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pce_summary(metric_long: pd.DataFrame, path: Path) -> None:
    pce = metric_long[(metric_long["comparison_family"] == "PCE") & metric_long["matched_trained"]].copy()
    metrics = [m for m in ["oof_r2", "oof_rmse", "oof_mae", "group_holdout_r2", "group_holdout_rmse"] if m in set(pce["metric"])]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4.2), dpi=180)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        row = pce[pce["metric"] == metric].iloc[0]
        ax.bar(["Before", "After"], [row["before"], row["after"]], color=["#6c8ebf", "#57a773"])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.5, 0.98, f"signed improvement\n{row['signed_improvement']:+.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.suptitle("PCE model performance before vs after literature-mined dataset update")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_support(aligned: pd.DataFrame, path: Path, limit: int = 22) -> None:
    data = aligned[aligned["matched_trained"] & aligned["before_n"].notna() & aligned["after_n"].notna()].copy()
    if data.empty:
        return
    data["label"] = data.apply(short_label, axis=1)
    data = data.sort_values("after_n", ascending=False).head(limit)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(data) + 1.8)), dpi=180)
    ax.barh(y - 0.18, data["before_n"], height=0.34, label="Before")
    ax.barh(y + 0.18, data["after_n"], height=0.34, label="After")
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Rows available for model target")
    ax.set_title("Training/evaluation support before vs after literature update")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_signed_improvement(metric_long: pd.DataFrame, path: Path, limit: int = 24) -> None:
    data = metric_long[metric_long["matched_trained"]].copy()
    if data.empty:
        return
    preferred = ["oof_r2", "oof_roc_auc", "oof_balanced_accuracy", "group_holdout_r2"]
    data = data[data["metric"].isin(preferred)]
    if data.empty:
        return
    data["label"] = data.apply(lambda r: f"{short_label(r)} | {r['metric']}", axis=1)
    data = data.reindex(data["signed_improvement"].abs().sort_values(ascending=False).index).head(limit)
    colors = np.where(data["signed_improvement"] >= 0, "#57a773", "#c85c5c")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.4 * len(data) + 1.8)), dpi=180)
    ax.barh(np.arange(len(data)), data["signed_improvement"], color=colors)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["label"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signed improvement after update (positive is better)")
    ax.set_title("Matched model performance change after literature-mined dataset update")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    before_label: str,
    after_label: str,
    aligned: pd.DataFrame,
    metric_long: pd.DataFrame,
    settings: pd.DataFrame,
) -> None:
    matched = aligned[aligned["matched_trained"]]
    config_mismatches = settings[~settings["matches"]] if not settings.empty and "matches" in settings.columns else settings
    improved = metric_long[metric_long["matched_trained"] & (metric_long["signed_improvement"] > 0)]
    worsened = metric_long[metric_long["matched_trained"] & (metric_long["signed_improvement"] < 0)]
    lines = [
        "Before/After Literature-Mining Model Comparison",
        "================================================",
        "",
        f"Before label: {before_label}",
        f"After label:  {after_label}",
        f"Matched trained model targets: {len(matched)}",
        f"Matched metric values improved: {len(improved)}",
        f"Matched metric values worsened: {len(worsened)}",
        f"Configuration/environment mismatches: {len(config_mismatches)}",
        "",
        "Interpretation safeguards:",
        "- Compare only identical model family, mode, target, task, code, backend, CV, and hyperparameter settings.",
        "- A larger target sample count is evidence that literature mining increased coverage.",
        "- A higher score suggests improved predictive performance, but should be confirmed on a frozen external holdout.",
        "- Do not average R2, RMSE, ROC-AUC, and classification accuracy into one scientific score.",
        "",
        "Open first:",
        "- plots/overview/signed_metric_improvement.png",
        "- plots/overview/target_support_before_after.png",
        "- plots/pce/pce_before_after_metrics.png",
        "- plots/stability_regression/stability_regression_r2_before_after.png",
        "- plots/stability_classification/stability_classification_auc_before_after.png",
        "- tables/aligned_model_comparison.csv",
        "- tables/configuration_comparison.csv",
    ]
    if len(config_mismatches):
        lines.extend(["", "WARNING: The runs have configuration/environment differences. Review configuration_comparison.csv before attributing changes to literature mining."])
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched PCE/stability model runs before and after a literature-mined dataset update.")
    parser.add_argument("--before", required=True, help="Baseline model output folder")
    parser.add_argument("--after", required=True, help="Literature-updated model output folder")
    parser.add_argument("--out", required=True, help="Comparison output folder")
    parser.add_argument("--before-label", default="Baseline database")
    parser.add_argument("--after-label", default="Literature-updated database")
    args = parser.parse_args()

    before_root, after_root, out_dir = Path(args.before), Path(args.after), ensure_dir(args.out)
    tables_dir = ensure_dir(out_dir / "tables")
    plots_dir = ensure_dir(out_dir / "plots")
    pce_plots = ensure_dir(plots_dir / "pce")
    regression_plots = ensure_dir(plots_dir / "stability_regression")
    classification_plots = ensure_dir(plots_dir / "stability_classification")
    overview_plots = ensure_dir(plots_dir / "overview")

    before, before_cfg, before_env = load_run(before_root, args.before_label)
    after, after_cfg, after_env = load_run(after_root, args.after_label)
    settings = compare_settings(before_cfg, after_cfg, before_env, after_env)
    aligned = make_aligned_table(before, after)
    metric_long = make_metric_long(aligned)

    settings.to_csv(tables_dir / "configuration_comparison.csv", index=False)
    aligned.to_csv(tables_dir / "aligned_model_comparison.csv", index=False)
    metric_long.to_csv(tables_dir / "metric_comparison_long.csv", index=False)
    aligned[aligned["_merge"] != "both"].to_csv(tables_dir / "unmatched_models.csv", index=False)

    plot_pce_summary(metric_long, pce_plots / "pce_before_after_metrics.png")
    paired_bar_plot(metric_long[metric_long["comparison_family"] != "PCE"], "oof_r2", "Stability regression OOF R2 before vs after", regression_plots / "stability_regression_r2_before_after.png")
    paired_bar_plot(metric_long[metric_long["comparison_family"] != "PCE"], "oof_rmse", "Stability regression OOF RMSE before vs after", regression_plots / "stability_regression_rmse_before_after.png")
    paired_bar_plot(metric_long[metric_long["comparison_family"] != "PCE"], "oof_roc_auc", "Stability classification OOF ROC-AUC before vs after", classification_plots / "stability_classification_auc_before_after.png")
    paired_bar_plot(metric_long[metric_long["comparison_family"] != "PCE"], "oof_balanced_accuracy", "Stability classification balanced accuracy before vs after", classification_plots / "stability_classification_balanced_accuracy_before_after.png")
    plot_support(aligned, overview_plots / "target_support_before_after.png")
    plot_signed_improvement(metric_long, overview_plots / "signed_metric_improvement.png")
    write_summary(out_dir, args.before_label, args.after_label, aligned, metric_long, settings)
    print(f"Comparison outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
