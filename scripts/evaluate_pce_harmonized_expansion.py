from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "data_provenance"
    / "datasets"
)
FROZEN_DIR = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "frozen_holdout_v1"
)
UTILITY_DIR = FROZEN_DIR.parent / "expansion_utility_audit_v1"
OUT_DIR = FROZEN_DIR.parent / "harmonized_expansion_v1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate schema-harmonized LiteratureAgent expansion on two domains."
    )
    parser.add_argument(
        "--original-csv", type=Path, default=DATA_DIR / "original_perovskite_database.csv"
    )
    parser.add_argument(
        "--curated-csv", type=Path, default=DATA_DIR / "google_drive_all_records.csv"
    )
    parser.add_argument(
        "--integrated-csv", type=Path, default=DATA_DIR / "final_integrated_database.csv"
    )
    parser.add_argument(
        "--model-script",
        type=Path,
        default=ROOT / "models" / "pce_then_stability_same_approach.py",
    )
    parser.add_argument("--frozen-dir", type=Path, default=FROZEN_DIR)
    parser.add_argument("--utility-dir", type=Path, default=UTILITY_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def canonical_value(value: object, column: str) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().strip("'\"")
    if not text:
        return np.nan
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                text = "; ".join(str(item).strip() for item in parsed if str(item).strip())
        except (SyntaxError, ValueError):
            pass
    text = re.sub(r"\s+", " ", text).strip()
    if "stack_sequence" in column.lower():
        text = re.sub(r"\s*/\s*", " | ", text)
        text = re.sub(r"\s*\|\s*", " | ", text)
    if any(token in column.lower() for token in ("compounds", "solvents", "ions")):
        text = re.sub(r"\s*;\s*", "; ", text)
    text = text.casefold()
    if column == "Cell_architecture":
        compact = re.sub(r"[^a-z0-9]", "", text)
        if compact in {"pin", "inverted", "invertedplanar", "invertedplanarstructure"}:
            return "pin"
        if compact in {"nip", "regular", "regularplanar", "regularplanarstructure"}:
            return "nip"
    text = text.replace("spiro-ometad", "spiro-meotad")
    return text


def harmonize(
    df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    result = df.copy()
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in categorical_columns:
        result[column] = result[column].map(lambda value: canonical_value(value, column))
    return result


def frozen_test(
    original: pd.DataFrame,
    prediction_path: Path,
    frozen,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    predictions = pd.read_csv(prediction_path)
    keys, _ = frozen.build_row_keys(
        original, group_col, ["Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]
    )
    by_key = pd.Series(original.index, index=keys)
    indices = by_key.reindex(predictions["row_key"]).to_numpy()
    if pd.isna(indices).any():
        raise RuntimeError(f"Could not recover all frozen rows from {prediction_path.name}.")
    test = original.loc[indices.astype(int)].copy()
    if "row_random" in prediction_path.name:
        blocked = set(predictions["row_key"].astype("string"))
        train_mask = ~keys.astype("string").isin(blocked)
    else:
        blocked = set(predictions["group"].astype("string"))
        train_mask = ~original[group_col].astype("string").isin(blocked)
    return original.loc[train_mask].copy(), test, keys, predictions


def source_cohorts(
    integrated: pd.DataFrame,
    curated: pd.DataFrame,
    frozen,
    utility,
    group_col: str,
) -> tuple[pd.Series, pd.Series]:
    identity = ["Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]
    integrated_keys, _ = frozen.build_row_keys(integrated, group_col, identity)
    curated_keys, _ = frozen.build_row_keys(curated, group_col, identity)
    source = utility.truthy(integrated["_source_added_by_literature_agent"])
    accepted = utility.normalized_text(
        integrated["_lit_agent_confidence_status"]
    ).eq("accepted")
    curated_match = integrated_keys.astype("string").isin(set(curated_keys.astype("string")))
    cohort = pd.Series("original", index=integrated.index, dtype="string")
    cohort.loc[source & accepted & curated_match] = "curated_added"
    cohort.loc[source & accepted & ~curated_match] = "web_added"
    return cohort, integrated_keys


def original_domain_models(
    original: pd.DataFrame,
    integrated: pd.DataFrame,
    cohort: pd.Series,
    integrated_keys: pd.Series,
    features: list[str],
    target: str,
    group_col: str,
    utility,
    frozen,
    model,
    cfg,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "doi_grouped": args.frozen_dir / "01_original_database_frozen_predictions.csv",
        "row_random": args.frozen_dir
        / "01_original_database_frozen_row_random_predictions.csv",
    }
    variants = {
        "harmonized_original": cohort.eq("never"),
        "harmonized_curated": cohort.eq("curated_added"),
        "harmonized_all_accepted": cohort.isin(["curated_added", "web_added"]),
    }
    metric_rows = []
    delta_rows = []
    for split, prediction_path in paths.items():
        original_train, test, _, original_predictions = frozen_test(
            original, prediction_path, frozen, group_col
        )
        if split == "doi_grouped":
            blocked_groups = set(original_predictions["group"].astype("string"))
            eligible_added = ~integrated[group_col].astype("string").isin(blocked_groups)
        else:
            blocked_keys = set(original_predictions["row_key"].astype("string"))
            eligible_added = ~integrated_keys.astype("string").isin(blocked_keys)
        for variant, mask in variants.items():
            added_train = integrated.loc[mask & eligible_added].copy()
            predicted = utility.train_variant(
                model,
                cfg,
                features,
                original_train,
                added_train,
                test,
                target,
                1.0,
            )
            actual = pd.to_numeric(test[target], errors="raise").to_numpy()
            metrics = utility.metric_values(actual, predicted)
            metric_rows.append(
                {
                    "evaluation_domain": "original_database",
                    "split_strategy": split,
                    "variant": variant,
                    "original_train_rows": len(original_train),
                    "added_train_rows": len(added_train),
                    "test_rows": len(test),
                    **metrics,
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "group": original_predictions["group"],
                    "row_key": original_predictions["row_key"],
                }
            )
            prediction_frame.to_csv(
                args.out_dir / f"original_domain_{split}_{variant}_predictions.csv",
                index=False,
            )
            for row in utility.bootstrap_paired_deltas(
                original_predictions,
                prediction_frame,
                args.bootstrap_iterations,
                args.random_state,
            ):
                delta_rows.append(
                    {
                        "evaluation_domain": "original_database",
                        "split_strategy": split,
                        "variant": variant,
                        **row,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def expanded_domain_models(
    original: pd.DataFrame,
    integrated: pd.DataFrame,
    cohort: pd.Series,
    features: list[str],
    target: str,
    group_col: str,
    utility,
    frozen,
    model,
    cfg,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    added = integrated.loc[cohort.isin(["curated_added", "web_added"])].copy()
    groups = added[group_col].astype("string").fillna("missing_group")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    train_index, test_index = next(splitter.split(added, groups=groups))
    added_train = added.iloc[train_index].copy()
    test = added.iloc[test_index].copy()
    test_groups = set(test[group_col].astype("string"))
    original_train = original.loc[
        ~original[group_col].astype("string").isin(test_groups)
    ].copy()
    test_keys, _ = frozen.build_row_keys(
        test, group_col, ["Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]
    )
    variants = {
        "original_only": integrated.iloc[0:0].copy(),
        "original_plus_accepted_expansion": added_train,
    }
    metric_rows = []
    predictions_by_variant = {}
    for variant, expansion_train in variants.items():
        predicted = utility.train_variant(
            model,
            cfg,
            features,
            original_train,
            expansion_train,
            test,
            target,
            1.0,
        )
        actual = pd.to_numeric(test[target], errors="raise").to_numpy()
        metrics = utility.metric_values(actual, predicted)
        metric_rows.append(
            {
                "evaluation_domain": "accepted_literatureagent",
                "split_strategy": "doi_grouped",
                "variant": variant,
                "original_train_rows": len(original_train),
                "added_train_rows": len(expansion_train),
                "test_rows": len(test),
                "test_doi_groups": len(test_groups),
                **metrics,
            }
        )
        frame = pd.DataFrame(
            {
                "actual": actual,
                "predicted": predicted,
                "group": test[group_col].astype("string").to_numpy(),
                "row_key": test_keys.astype("string").to_numpy(),
            }
        )
        predictions_by_variant[variant] = frame
        frame.to_csv(args.out_dir / f"expanded_domain_{variant}_predictions.csv", index=False)
    baseline = predictions_by_variant["original_only"]
    revised = predictions_by_variant["original_plus_accepted_expansion"]
    deltas = utility.bootstrap_paired_deltas(
        baseline,
        revised,
        args.bootstrap_iterations,
        args.random_state,
    )
    delta_table = pd.DataFrame(
        [
            {
                "evaluation_domain": "accepted_literatureagent",
                "split_strategy": "doi_grouped",
                "variant": "original_plus_accepted_expansion",
                **row,
            }
            for row in deltas
        ]
    )
    return pd.DataFrame(metric_rows), delta_table


def make_figure(
    cohort_summary: pd.DataFrame,
    existing: pd.DataFrame,
    original_metrics: pd.DataFrame,
    expanded_metrics: pd.DataFrame,
    path_png: Path,
    path_pdf: Path,
) -> None:
    summary = cohort_summary.set_index("cohort")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    cohorts = ["original", "all_added"]
    labels = ["Original", "Accepted\nexpansion"]
    colors = ["#4C78A8", "#E69F00"]
    completeness = [summary.loc[item, "feature_completeness_median"] for item in cohorts]
    axes[0, 0].bar(labels, completeness, color=colors, edgecolor="#222222")
    axes[0, 0].set_ylabel("Fraction of 278 predictors populated")
    axes[0, 0].set_ylim(0, max(completeness) * 1.25)
    axes[0, 0].set_title("A  Model-feature completeness")
    pce = [summary.loc[item, "pce_median"] for item in cohorts]
    axes[0, 1].bar(labels, pce, color=colors, edgecolor="#222222")
    axes[0, 1].set_ylabel("Median PCE (%)")
    axes[0, 1].set_ylim(0, max(pce) * 1.25)
    axes[0, 1].set_title("B  Cohort target shift")

    fixed = existing.loc[existing["split_strategy"].eq("doi_grouped")].set_index(
        "dataset_version"
    )
    harmonized = original_metrics.loc[
        original_metrics["split_strategy"].eq("doi_grouped")
    ].set_index("variant")
    fixed_values = [
        fixed.loc["01_original_database", "r2"],
        fixed.loc["03_after_google_drive_plus_expansion", "r2"],
        harmonized.loc["harmonized_original", "r2"],
        harmonized.loc["harmonized_all_accepted", "r2"],
    ]
    fixed_labels = ["Original", "Raw\nintegration", "Harmonized\noriginal", "Harmonized\nexpansion"]
    axes[1, 0].bar(
        fixed_labels,
        fixed_values,
        color=["#4C78A8", "#D55E00", "#72B7B2", "#59A14F"],
        edgecolor="#222222",
    )
    axes[1, 0].axhline(fixed_values[0], color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("R²")
    axes[1, 0].set_title("C  Fixed original-domain DOI holdout")

    expanded = expanded_metrics.set_index("variant")
    expanded_values = [
        expanded.loc["original_only", "r2"],
        expanded.loc["original_plus_accepted_expansion", "r2"],
    ]
    axes[1, 1].bar(
        ["Original-only\ntraining", "Original + accepted\nexpansion"],
        expanded_values,
        color=["#4C78A8", "#59A14F"],
        edgecolor="#222222",
    )
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_title("D  DOI-disjoint LiteratureAgent-domain holdout")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=9)
    fig.suptitle(
        "Structured literature expansion: integration audit and domain-specific utility",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    utility = load_module(
        ROOT / "scripts" / "analyze_pce_expansion_utility.py", "pce_utility_helpers"
    )
    frozen = load_module(
        ROOT / "scripts" / "compare_pce_frozen_production_extratrees.py",
        "pce_frozen_helpers",
    )
    model = frozen.load_module(args.model_script)
    cfg = model.Config(
        INPUT_CSV=str(args.original_csv),
        OUTPUT_DIR=str(args.out_dir / "preparation_audit"),
        PCE_ONLY=True,
        N_ESTIMATORS=args.n_estimators,
        MODEL_BACKEND="extra_trees",
        RANDOM_STATE=args.random_state,
    )
    original, target, features, _, group_col = frozen.prepare(
        args.original_csv, model, cfg, args.out_dir / "preparation_audit" / "original"
    )
    integrated, _, _, _, _ = frozen.prepare(
        args.integrated_csv,
        model,
        cfg,
        args.out_dir / "preparation_audit" / "integrated",
        feature_cols=features,
    )
    curated, _, _, _, _ = frozen.prepare(
        args.curated_csv,
        model,
        cfg,
        args.out_dir / "preparation_audit" / "curated",
        feature_cols=features,
    )
    cohort, integrated_keys = source_cohorts(
        integrated, curated, frozen, utility, group_col
    )
    numeric_columns, categorical_columns = model.split_num_cat(original[features], cfg)
    original = harmonize(original, numeric_columns, categorical_columns)
    integrated = harmonize(integrated, numeric_columns, categorical_columns)
    curated = harmonize(curated, numeric_columns, categorical_columns)

    original_metrics, original_deltas = original_domain_models(
        original,
        integrated,
        cohort,
        integrated_keys,
        features,
        target,
        group_col,
        utility,
        frozen,
        model,
        cfg,
        args,
    )
    expanded_metrics, expanded_deltas = expanded_domain_models(
        original,
        integrated,
        cohort,
        features,
        target,
        group_col,
        utility,
        frozen,
        model,
        cfg,
        args,
    )
    metrics = pd.concat([original_metrics, expanded_metrics], ignore_index=True)
    deltas = pd.concat([original_deltas, expanded_deltas], ignore_index=True)
    metrics.to_csv(args.out_dir / "harmonized_domain_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "harmonized_domain_paired_deltas.csv", index=False)
    existing = pd.read_csv(args.frozen_dir / "frozen_production_pce_comparison.csv")
    cohort_summary = pd.read_csv(args.utility_dir / "model_eligible_cohort_summary.csv")
    make_figure(
        cohort_summary,
        existing,
        original_metrics,
        expanded_metrics,
        args.out_dir / "figure_pce_harmonized_domain_evaluation.png",
        args.out_dir / "figure_pce_harmonized_domain_evaluation.pdf",
    )
    report = [
        "# Harmonized PCE Expansion Evaluation",
        "",
        "## Fixed original-database holdouts",
        "",
        utility.markdown_table(original_metrics),
        "",
        "## DOI-disjoint accepted-LiteratureAgent holdout",
        "",
        utility.markdown_table(expanded_metrics),
        "",
        "## Paired DOI-cluster bootstrap deltas",
        "",
        utility.markdown_table(deltas, digits=4),
        "",
        "## Evaluation boundary",
        "",
        "- Original-domain tests measure backward compatibility with the historical database.",
        "- LiteratureAgent-domain tests measure whether accepted expansion records improve prediction for held-out accepted expansion records.",
        "- All test DOI groups are excluded from both original and expansion training rows.",
        "- Unknown boolean values remain unknown; they are not converted to false.",
    ]
    (args.out_dir / "HARMONIZED_EXPANSION_EVALUATION.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    manifest = {
        "feature_count": len(features),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "n_estimators": args.n_estimators,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_state": args.random_state,
        "expanded_domain_test_size": args.test_size,
        "boolean_unknown_policy": "preserve_missing",
    }
    (args.out_dir / "harmonized_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nPAIRED DELTAS")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
