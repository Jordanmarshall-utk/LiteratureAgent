#!/usr/bin/env python3
"""Run leakage-controlled PCE comparisons with identical data and CV settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BACKENDS = ("extra_trees", "xgboost")


def stream_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            console_line = line.encode(console_encoding, errors="replace").decode(
                console_encoding, errors="replace"
            )
            print(console_line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Model command failed with exit code {return_code}. See {log_path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_fingerprint(path: Path) -> str:
    frame = pd.read_csv(path, low_memory=False)
    source_cols = [column for column in frame.columns if column != "oof_pred"]
    normalized = frame[source_cols].fillna("<NA>").astype(str)
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_metrics(backend: str, family: str, metrics: dict) -> dict:
    params = metrics.get("estimator_params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params.replace("'", '"'))
        except Exception:
            params = {"raw": params}
    return {
        "backend": backend,
        "model_family": family,
        "status": metrics.get("status"),
        "n": metrics.get("n"),
        "n_features_raw": metrics.get("n_features_raw"),
        "cv_strategy": metrics.get("cv_strategy"),
        "oof_r2": metrics.get("oof_r2"),
        "oof_rmse": metrics.get("oof_rmse"),
        "oof_mae": metrics.get("oof_mae"),
        "group_holdout_r2": metrics.get("group_holdout_r2"),
        "group_holdout_rmse": metrics.get("group_holdout_rmse"),
        "group_holdout_mae": metrics.get("group_holdout_mae"),
        "estimator_class": metrics.get("estimator_class"),
        "actual_n_estimators": metrics.get("actual_n_estimators"),
        "estimator_params_json": json.dumps(params, sort_keys=True),
    }


def make_plot(comparison: pd.DataFrame, output_path: Path, cv_strategy: str) -> None:
    direct = comparison[comparison["model_family"].eq("pce_direct")].copy()
    if direct.empty:
        return
    direct = direct.set_index("backend").reindex(BACKENDS)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=220)
    colors = ["#4C78A8", "#E45756"]

    direct["oof_r2"].plot(kind="bar", ax=axes[0], color=colors, width=0.68)
    evaluation_label = "DOI-grouped" if cv_strategy == "grouped" else "row-random diagnostic"
    axes[0].set_title(f"{evaluation_label} out-of-fold PCE performance")
    axes[0].set_ylabel("R²")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].axhline(0, color="black", linewidth=0.8)
    for patch, value in zip(axes[0].patches, direct["oof_r2"]):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    errors = direct[["oof_rmse", "oof_mae"]].rename(
        columns={"oof_rmse": "RMSE", "oof_mae": "MAE"}
    )
    errors.plot(kind="bar", ax=axes[1], color=["#72B7B2", "#F2CF5B"], width=0.72)
    axes[1].set_title(f"{evaluation_label} out-of-fold PCE errors")
    axes[1].set_ylabel("PCE percentage points")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"PCE regressor comparison on identical rows and features ({evaluation_label})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Integrated Perovskite Database CSV")
    parser.add_argument(
        "--out",
        required=True,
        help="Comparison output directory; backend-specific runs are written below it",
    )
    parser.add_argument(
        "--model-script",
        default=str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "pce_then_stability_same_approach.py"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=650)
    parser.add_argument("--min-publication-year", type=int, default=2018)
    parser.add_argument(
        "--cv-strategy",
        choices=["grouped", "row_random"],
        default="grouped",
        help="Grouped is production evaluation; row_random is an explicitly labeled diagnostic.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip the hierarchical residual model for a faster direct-model comparison",
    )
    args = parser.parse_args()

    output_root = Path(args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_script = Path(args.model_script).resolve()
    input_csv = Path(args.csv).resolve()

    for backend in BACKENDS:
        backend_out = output_root / backend
        command = [
            sys.executable,
            "-u",
            str(model_script),
            "--csv",
            str(input_csv),
            "--out",
            str(backend_out),
            "--model-backend",
            backend,
            "--n-estimators",
            str(args.n_estimators),
            "--min-publication-year",
            str(args.min_publication_year),
            "--cv-strategy",
            args.cv_strategy,
            "--pce-only",
        ]
        if args.direct_only:
            command.append("--disable-pce-hierarchical-residual")
        stream_command(command, output_root / f"{backend}_run.log")

    fingerprints = {
        backend: prediction_fingerprint(
            output_root / backend / "pce" / "pce_direct_oof_predictions.csv"
        )
        for backend in BACKENDS
    }
    identical_inputs = len(set(fingerprints.values())) == 1
    if not identical_inputs:
        raise RuntimeError(
            "Backend runs did not use identical ordered targets/groups. "
            f"Fingerprints: {fingerprints}"
        )

    rows = []
    for backend in BACKENDS:
        backend_out = output_root / backend
        direct = load_json(backend_out / "pce" / "pce_direct_metrics.json")
        rows.append(flatten_metrics(backend, "pce_direct", direct))
        hierarchical_path = (
            backend_out
            / "pce"
            / "hierarchical_residual"
            / "pce_hierarchical_residual_metrics.json"
        )
        if hierarchical_path.exists():
            rows.append(
                flatten_metrics(
                    backend,
                    "pce_hierarchical_residual",
                    load_json(hierarchical_path),
                )
            )

    comparison = pd.DataFrame(rows)
    comparison["identical_input_fingerprint"] = identical_inputs
    comparison["input_fingerprint_sha256"] = fingerprints[BACKENDS[0]]
    comparison.to_csv(output_root / "pce_regressor_comparison.csv", index=False)
    make_plot(
        comparison,
        output_root / "pce_regressor_comparison.png",
        args.cv_strategy,
    )

    direct = comparison[comparison["model_family"].eq("pce_direct")].sort_values(
        "oof_r2", ascending=False
    )
    winner = direct.iloc[0]
    summary = {
        "input_csv": str(input_csv),
        "model_script": str(model_script),
        "n_estimators_requested": args.n_estimators,
        "min_publication_year": args.min_publication_year,
        "cv_strategy": args.cv_strategy,
        "identical_ordered_targets_and_groups": identical_inputs,
        "input_fingerprint_sha256": fingerprints[BACKENDS[0]],
        "winning_backend_by_grouped_oof_r2": winner["backend"],
        "winning_grouped_oof_r2": winner["oof_r2"],
        "comparison_csv": str(output_root / "pce_regressor_comparison.csv"),
        "comparison_plot": str(output_root / "pce_regressor_comparison.png"),
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nPCE regressor comparison complete")
    print(comparison.to_string(index=False))
    print(
        f"\nWinner by {args.cv_strategy} OOF R²: "
        f"{winner['backend']} ({winner['oof_r2']:.4f})"
    )


if __name__ == "__main__":
    main()
