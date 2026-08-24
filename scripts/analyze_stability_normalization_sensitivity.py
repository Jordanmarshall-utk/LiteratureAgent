from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    r"E:\LiteratureAgent\final_model_comparison_fast_20260811\grouped\integrated\stability\physical_layer\physical_targets_preview.csv"
)
DEFAULT_OUT = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "stability_normalization_sensitivity_v1"
)
DEFAULT_FIGURE_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "publication_figures"
    / "main"
)

SCENARIOS = [
    ("Ea 0.30 eV", 0.30, 1.0, 0.7),
    ("Ea 0.45 eV", 0.45, 1.0, 0.7),
    ("Baseline", 0.60, 1.0, 0.7),
    ("Ea 0.75 eV", 0.75, 1.0, 0.7),
    ("Ea 0.90 eV", 0.90, 1.0, 0.7),
    ("RH exponent 0.5", 0.60, 0.5, 0.7),
    ("RH exponent 2.0", 0.60, 2.0, 0.7),
    ("Light exponent 0.5", 0.60, 1.0, 0.5),
    ("Light exponent 1.0", 0.60, 1.0, 1.0),
]


def calculate(
    frame: pd.DataFrame,
    *,
    activation_energy: float,
    humidity_exponent: float,
    light_exponent: float,
    reference_temp_k: float = 300.0,
    reference_rh: float = 20.0,
    reference_light: float = 1.0,
) -> pd.DataFrame:
    kb_ev_k = 8.617333262e-5
    temp = pd.to_numeric(frame["PHYS_test_temp_K"], errors="coerce")
    rh = pd.to_numeric(frame["PHYS_test_RH_percent"], errors="coerce")
    light = pd.to_numeric(frame["PHYS_test_light_sun"], errors="coerce")
    k_obs = pd.to_numeric(frame["PHYS_k_obs_h_inv"], errors="coerce")

    exponent = (activation_energy / kb_ev_k) * (
        (1.0 / reference_temp_k) - (1.0 / temp)
    )
    af_temp = np.exp(np.clip(exponent, -50, 50))
    af_rh = np.where(rh <= reference_rh, 1.0, (rh / reference_rh) ** humidity_exponent)
    af_light = (light / reference_light) ** light_exponent
    af_total = af_temp * af_rh * af_light
    k_ref = k_obs / af_total
    t80_ref = -np.log(0.8) / k_ref

    output = pd.DataFrame(
        {
            "af_total": af_total,
            "k_ref": k_ref,
            "t80_ref": t80_ref,
        }
    )
    valid = (
        np.isfinite(output["af_total"])
        & np.isfinite(output["k_ref"])
        & np.isfinite(output["t80_ref"])
        & output["af_total"].gt(0)
        & output["k_ref"].gt(0)
        & output["t80_ref"].gt(0)
    )
    return output.where(valid)


def analyze(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for label, ea, rh_exp, light_exp in SCENARIOS:
        results[label] = calculate(
            frame,
            activation_energy=ea,
            humidity_exponent=rh_exp,
            light_exponent=light_exp,
        )

    baseline = results["Baseline"]
    valid_baseline = baseline["t80_ref"].notna()
    baseline_rank = baseline.loc[valid_baseline, "t80_ref"]
    baseline_top = set(baseline_rank.nlargest(max(1, len(baseline_rank) // 4)).index)
    rows: list[dict[str, object]] = []
    long_rows: list[pd.DataFrame] = []

    for label, ea, rh_exp, light_exp in SCENARIOS:
        values = results[label]
        common = valid_baseline & values["t80_ref"].notna()
        rho = spearmanr(
            baseline.loc[common, "t80_ref"], values.loc[common, "t80_ref"]
        ).statistic
        scenario_top = set(
            values.loc[common, "t80_ref"].nlargest(max(1, int(common.sum()) // 4)).index
        )
        overlap = len(baseline_top & scenario_top) / max(1, len(baseline_top))
        valid = values["t80_ref"].dropna()
        rows.append(
            {
                "scenario": label,
                "activation_energy_eV": ea,
                "humidity_exponent": rh_exp,
                "light_exponent": light_exp,
                "valid_rows": int(len(valid)),
                "median_AF_total": float(values["af_total"].median()),
                "median_T80_ref_h": float(valid.median()),
                "T80_ref_q25_h": float(valid.quantile(0.25)),
                "T80_ref_q75_h": float(valid.quantile(0.75)),
                "spearman_vs_baseline": float(rho),
                "top_quartile_overlap_vs_baseline": float(overlap),
            }
        )
        subset = np.log10(valid).replace([np.inf, -np.inf], np.nan).dropna()
        long_rows.append(
            pd.DataFrame({"scenario": label, "log10_T80_ref_h": subset.to_numpy()})
        )
    return pd.DataFrame(rows), pd.concat(long_rows, ignore_index=True)


def plot(summary: pd.DataFrame, long_values: pd.DataFrame, stem: Path) -> None:
    order = [item[0] for item in SCENARIOS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180, gridspec_kw={"width_ratios": [1.55, 1]})

    data = [
        long_values.loc[long_values["scenario"].eq(label), "log10_T80_ref_h"].to_numpy()
        for label in order
    ]
    box = axes[0].boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch, label in zip(box["boxes"], order):
        patch.set_facecolor("#0F766E" if label == "Baseline" else "#D9D9D9")
        patch.set_edgecolor("#333333")
    axes[0].tick_params(axis="x", rotation=58, labelsize=8.5)
    axes[0].set_ylabel("log10 normalized T80 (h)")
    axes[0].set_title("Normalized target distribution", fontweight="bold")
    axes[0].grid(axis="y", color="#E5E5E5", linewidth=0.7)

    y = np.arange(len(order))
    corr = summary.set_index("scenario").loc[order, "spearman_vs_baseline"]
    colors = ["#0F766E" if label == "Baseline" else "#777777" for label in order]
    axes[1].barh(y, corr, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y, order, fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.04)
    axes[1].set_xlabel("Spearman rho vs baseline")
    axes[1].set_title("Rank sensitivity", fontweight="bold")
    axes[1].grid(axis="x", color="#E5E5E5", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Sensitivity of semi-empirical stability normalization",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "One-at-a-time parameter changes; this evaluates target normalization, not mechanistic validity.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(
            stem.with_suffix(suffix),
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, low_memory=False)
    summary, long_values = analyze(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "stability_normalization_sensitivity.csv", index=False)
    long_values.to_csv(args.out_dir / "stability_normalized_targets_long.csv", index=False)
    stem = args.figure_dir / "figure_stability_normalization_sensitivity"
    plot(summary, long_values, stem)
    manifest = {
        "input_csv": str(args.input_csv),
        "scenarios": len(SCENARIOS),
        "baseline": {
            "activation_energy_eV": 0.60,
            "humidity_exponent": 1.0,
            "light_exponent": 0.7,
            "reference_temperature_K": 300.0,
            "reference_RH_percent": 20.0,
            "reference_light_sun": 1.0,
        },
        "scope": "normalization_target_sensitivity_not_mechanistic_validation",
        "figure": str(stem.with_suffix(".png")),
    }
    (args.out_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
