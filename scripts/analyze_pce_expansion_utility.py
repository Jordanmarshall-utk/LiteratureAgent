from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "data_provenance"
    / "datasets"
)
DEFAULT_FROZEN = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "frozen_holdout_v1"
)
DEFAULT_OUT = (
    ROOT
    / "artifacts"
    / "literature_agent_pce_stability"
    / "source_data"
    / "model_results"
    / "expansion_utility_audit_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether LiteratureAgent expansion improves model-ready PCE utility."
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=DEFAULT_DATA / "original_perovskite_database.csv",
    )
    parser.add_argument(
        "--google-drive-csv",
        type=Path,
        default=DEFAULT_DATA / "google_drive_all_records.csv",
    )
    parser.add_argument(
        "--integrated-csv",
        type=Path,
        default=DEFAULT_DATA / "final_integrated_database.csv",
    )
    parser.add_argument(
        "--model-script",
        type=Path,
        default=ROOT / "models" / "pce_then_stability_same_approach.py",
    )
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def truthy(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "1.0", "true", "yes", "y"})
        .fillna(False)
    )


def normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def feature_completeness(df: pd.DataFrame, features: list[str]) -> pd.Series:
    count = np.zeros(len(df), dtype=float)
    missing_tokens = {"", "nan", "none", "null", "na", "n/a", "not available"}
    for column in features:
        values = df[column]
        present = values.notna()
        if values.dtype == "object" or isinstance(values.dtype, pd.StringDtype):
            text = values.astype("string").str.strip().str.lower()
            present &= ~text.isin(missing_tokens).fillna(True)
        count += present.to_numpy(dtype=float)
    return pd.Series(count / len(features), index=df.index, name="feature_completeness")


def fill_rates(df: pd.DataFrame, features: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=features, dtype=float)
    return pd.Series(
        {column: float(feature_completeness(df[[column]], [column]).mean()) for column in features}
    )


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def bootstrap_paired_deltas(
    original: pd.DataFrame,
    revised: pd.DataFrame,
    iterations: int,
    random_state: int,
) -> list[dict[str, float | str]]:
    merged = original.merge(
        revised[["row_key", "predicted"]],
        on="row_key",
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_revised"),
    )
    if len(merged) != len(original):
        raise RuntimeError("New predictions do not match every frozen original test row.")
    groups = merged["group"].astype("string").fillna("missing_group")
    unique_groups = groups.unique()
    group_indices = {
        group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups
    }
    rng = np.random.default_rng(random_state)
    draws = {"r2": [], "rmse": [], "mae": []}
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        actual = merged["actual"].to_numpy()[indices]
        pred_original = merged["predicted_original"].to_numpy()[indices]
        pred_revised = merged["predicted_revised"].to_numpy()[indices]
        original_metrics = metric_values(actual, pred_original)
        revised_metrics = metric_values(actual, pred_revised)
        for metric in draws:
            draws[metric].append(revised_metrics[metric] - original_metrics[metric])
    rows: list[dict[str, float | str]] = []
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "delta_vs_original": float(np.median(array)),
                "ci_low": float(np.quantile(array, 0.025)),
                "ci_high": float(np.quantile(array, 0.975)),
            }
        )
    return rows


def cohort_summary(
    df: pd.DataFrame,
    cohort: pd.Series,
    completeness: pd.Series,
    target: str,
    group_col: str,
) -> pd.DataFrame:
    rows = []
    for label in ["original", "curated_added", "web_added", "all_added"]:
        if label == "all_added":
            mask = cohort.isin(["curated_added", "web_added"])
        else:
            mask = cohort.eq(label)
        subset = df.loc[mask]
        values = pd.to_numeric(subset[target], errors="coerce")
        comp = completeness.loc[mask]
        rows.append(
            {
                "cohort": label,
                "model_eligible_rows": int(mask.sum()),
                "doi_groups": int(subset[group_col].astype("string").nunique()),
                "pce_mean": float(values.mean()),
                "pce_median": float(values.median()),
                "pce_std": float(values.std()),
                "pce_q25": float(values.quantile(0.25)),
                "pce_q75": float(values.quantile(0.75)),
                "feature_completeness_mean": float(comp.mean()),
                "feature_completeness_median": float(comp.median()),
                "feature_completeness_q25": float(comp.quantile(0.25)),
                "feature_completeness_q75": float(comp.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def raw_eligibility_summary(raw: pd.DataFrame, model, cfg, target: str) -> pd.DataFrame:
    source = truthy(raw["_source_added_by_literature_agent"])
    status = normalized_text(raw["_lit_agent_confidence_status"])
    pce = model.to_numeric_series(raw[target])
    years = model.infer_publication_years(raw)
    year_ok = years.isna() | years.ge(cfg.MIN_PUBLICATION_YEAR)
    rows = []
    cohorts = {
        "original": ~source,
        "accepted": source & status.eq("accepted"),
        "sparse_accepted": source & status.eq("sparse_accepted"),
    }
    for label, mask in cohorts.items():
        rows.append(
            {
                "cohort": label,
                "raw_rows": int(mask.sum()),
                "pce_nonmissing": int((mask & pce.notna()).sum()),
                "pce_below_5": int((mask & pce.notna() & pce.lt(cfg.PCE_MIN)).sum()),
                "pce_above_30": int((mask & pce.notna() & pce.gt(cfg.PCE_MAX)).sum()),
                "publication_year_eligible": int((mask & year_ok).sum()),
                "pce_and_year_eligible": int(
                    (mask & year_ok & pce.between(cfg.PCE_MIN, cfg.PCE_MAX)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def distribution_shift_tables(
    df: pd.DataFrame,
    cohort: pd.Series,
    features: list[str],
    model,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = df.loc[cohort.eq("original"), features]
    added = df.loc[cohort.isin(["curated_added", "web_added"]), features]
    curated = df.loc[cohort.eq("curated_added"), features]
    web = df.loc[cohort.eq("web_added"), features]
    rates = pd.DataFrame(
        {
            "feature": features,
            "original_fill_rate": fill_rates(baseline, features).reindex(features).to_numpy(),
            "curated_fill_rate": fill_rates(curated, features).reindex(features).to_numpy(),
            "web_fill_rate": fill_rates(web, features).reindex(features).to_numpy(),
            "all_added_fill_rate": fill_rates(added, features).reindex(features).to_numpy(),
        }
    )
    rates["added_minus_original"] = (
        rates["all_added_fill_rate"] - rates["original_fill_rate"]
    )
    rates["absolute_fill_rate_change"] = rates["added_minus_original"].abs()
    rates = rates.sort_values("absolute_fill_rate_change", ascending=False)

    numeric_rows = []
    categorical_rows = []
    for column in features:
        base_raw = baseline[column]
        added_raw = added[column]
        base_numeric = model.to_numeric_series(base_raw)
        added_numeric = model.to_numeric_series(added_raw)
        base_ratio = float(base_numeric.notna().sum() / max(base_raw.notna().sum(), 1))
        added_ratio = float(added_numeric.notna().sum() / max(added_raw.notna().sum(), 1))
        if min(base_ratio, added_ratio) >= 0.8 and min(
            base_numeric.notna().sum(), added_numeric.notna().sum()
        ) >= 30:
            a = base_numeric.dropna().to_numpy(dtype=float)
            b = added_numeric.dropna().to_numpy(dtype=float)
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            numeric_rows.append(
                {
                    "feature": column,
                    "original_n": len(a),
                    "added_n": len(b),
                    "original_mean": float(np.mean(a)),
                    "added_mean": float(np.mean(b)),
                    "standardized_mean_difference": float(
                        (np.mean(b) - np.mean(a)) / pooled if pooled > 0 else 0.0
                    ),
                    "ks_statistic": float(ks_2samp(a, b).statistic),
                }
            )
            continue
        a = normalized_text(base_raw)
        b = normalized_text(added_raw)
        a = a[a.ne("")]
        b = b[b.ne("")]
        if len(a) < 30 or len(b) < 30:
            continue
        base_levels = set(a.unique())
        unseen_rate = float((~b.isin(base_levels)).mean())
        categories = sorted(set(a.unique()).union(set(b.unique())))
        pa = a.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        pb = b.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        categorical_rows.append(
            {
                "feature": column,
                "original_n": len(a),
                "added_n": len(b),
                "original_unique": int(a.nunique()),
                "added_unique": int(b.nunique()),
                "added_unseen_category_rate": unseen_rate,
                "total_variation_distance": float(0.5 * np.abs(pa - pb).sum()),
            }
        )
    numeric = pd.DataFrame(numeric_rows)
    if not numeric.empty:
        numeric["absolute_smd"] = numeric["standardized_mean_difference"].abs()
        numeric = numeric.sort_values(["ks_statistic", "absolute_smd"], ascending=False)
    categorical = pd.DataFrame(categorical_rows)
    if not categorical.empty:
        categorical = categorical.sort_values(
            ["total_variation_distance", "added_unseen_category_rate"], ascending=False
        )
    return rates, numeric, categorical


def train_variant(
    model,
    cfg,
    features: list[str],
    original_train: pd.DataFrame,
    added_train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    added_weight: float,
) -> np.ndarray:
    train = pd.concat([original_train, added_train], ignore_index=True, sort=False)
    x_train = train[features].copy()
    y_train = pd.to_numeric(train[target], errors="raise")
    x_test = test[features].copy()
    preprocessor, numeric_columns, categorical_columns = model.make_preprocessor(
        x_train, cfg
    )
    for column in numeric_columns:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    for column in categorical_columns:
        x_train[column] = x_train[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
        x_test[column] = x_test[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
    estimator = model.make_model("regression", cfg)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    fit_kwargs = {}
    if added_weight != 1.0 and len(added_train):
        weights = np.concatenate(
            [
                np.ones(len(original_train), dtype=float),
                np.full(len(added_train), added_weight, dtype=float),
            ]
        )
        fit_kwargs["model__sample_weight"] = weights
    pipeline.fit(x_train, y_train, **fit_kwargs)
    return pipeline.predict(x_test)


def run_fixed_holdout_ablations(
    original: pd.DataFrame,
    integrated: pd.DataFrame,
    cohort: pd.Series,
    completeness: pd.Series,
    features: list[str],
    target: str,
    group_col: str,
    frozen,
    model,
    cfg,
    frozen_dir: Path,
    out_dir: Path,
    iterations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    identity_cols = ["Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]
    original_keys, _ = frozen.build_row_keys(original, group_col, identity_cols)
    integrated_keys, _ = frozen.build_row_keys(integrated, group_col, identity_cols)
    original_by_key = pd.Series(original.index, index=original_keys)
    accepted = normalized_text(integrated["_lit_agent_confidence_status"]).eq("accepted")
    added = cohort.isin(["curated_added", "web_added"]) & accepted
    high_threshold = float(completeness.loc[added].median())
    variants = {
        "web_added_only": {
            "mask": cohort.eq("web_added") & accepted,
            "weight": 1.0,
        },
        "accepted_top_half_completeness": {
            "mask": added & completeness.ge(high_threshold),
            "weight": 1.0,
        },
        "accepted_all_weight_0.5": {"mask": added, "weight": 0.5},
    }
    prediction_paths = {
        "doi_grouped": frozen_dir / "01_original_database_frozen_predictions.csv",
        "row_random": frozen_dir
        / "01_original_database_frozen_row_random_predictions.csv",
    }
    metric_rows = []
    delta_rows = []
    for split_strategy, prediction_path in prediction_paths.items():
        original_predictions = pd.read_csv(prediction_path)
        test_indices = original_by_key.reindex(original_predictions["row_key"]).to_numpy()
        if pd.isna(test_indices).any():
            raise RuntimeError(f"Could not recover frozen test rows for {split_strategy}.")
        test_indices = test_indices.astype(int)
        test = original.loc[test_indices].copy()
        if not np.allclose(
            pd.to_numeric(test[target], errors="coerce").to_numpy(),
            original_predictions["actual"].to_numpy(),
            equal_nan=False,
        ):
            raise RuntimeError(f"Frozen target mismatch for {split_strategy}.")
        if split_strategy == "doi_grouped":
            frozen_groups = set(original_predictions["group"].astype("string"))
            original_train_mask = ~original[group_col].astype("string").isin(frozen_groups)
            eligible_added = ~integrated[group_col].astype("string").isin(frozen_groups)
        else:
            frozen_keys = set(original_predictions["row_key"].astype("string"))
            original_train_mask = ~original_keys.astype("string").isin(frozen_keys)
            eligible_added = ~integrated_keys.astype("string").isin(frozen_keys)
        original_train = original.loc[original_train_mask].copy()
        for variant, definition in variants.items():
            variant_mask = definition["mask"] & eligible_added
            added_train = integrated.loc[variant_mask].copy()
            predictions = train_variant(
                model=model,
                cfg=cfg,
                features=features,
                original_train=original_train,
                added_train=added_train,
                test=test,
                target=target,
                added_weight=float(definition["weight"]),
            )
            actual = pd.to_numeric(test[target], errors="raise").to_numpy()
            metrics = metric_values(actual, predictions)
            metric_rows.append(
                {
                    "split_strategy": split_strategy,
                    "variant": variant,
                    "original_train_rows": len(original_train),
                    "added_train_rows": len(added_train),
                    "frozen_test_rows": len(test),
                    **metrics,
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "actual": actual,
                    "predicted": predictions,
                    "group": original_predictions["group"].to_numpy(),
                    "row_key": original_predictions["row_key"].to_numpy(),
                }
            )
            prediction_frame.to_csv(
                out_dir / f"{split_strategy}_{variant}_predictions.csv", index=False
            )
            for row in bootstrap_paired_deltas(
                original_predictions,
                prediction_frame,
                iterations=iterations,
                random_state=random_state,
            ):
                delta_rows.append(
                    {
                        "split_strategy": split_strategy,
                        "variant": variant,
                        **row,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows), high_threshold


def make_figure(
    cohort_table: pd.DataFrame,
    model_table: pd.DataFrame,
    path_png: Path,
    path_pdf: Path,
) -> None:
    labels = {
        "original": "Original",
        "curated_added": "Curated\nadded",
        "web_added": "Web\nadded",
        "all_added": "All\nadded",
        "01_original_database": "Original",
        "02_after_google_drive": "Curated",
        "03_after_google_drive_plus_expansion": "All added",
        "web_added_only": "Web only",
        "accepted_top_half_completeness": "Top-half\ncomplete",
        "accepted_all_weight_0.5": "All, 0.5x\nweight",
    }
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#A0A0A0"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    x = np.arange(len(cohort_table))
    med = cohort_table["feature_completeness_median"].to_numpy()
    low = med - cohort_table["feature_completeness_q25"].to_numpy()
    high = cohort_table["feature_completeness_q75"].to_numpy() - med
    axes[0, 0].bar(x, med, color=colors, edgecolor="#222222", linewidth=0.8)
    axes[0, 0].errorbar(x, med, yerr=np.vstack([low, high]), fmt="none", color="#222222")
    axes[0, 0].set_xticks(x, [labels[v] for v in cohort_table["cohort"]])
    axes[0, 0].set_ylabel("Fraction of 278 predictors populated")
    axes[0, 0].set_title("A  Predictor completeness")
    axes[0, 0].set_ylim(0, min(1.0, max(0.25, float((med + high).max()) * 1.2)))

    pce_med = cohort_table["pce_median"].to_numpy()
    pce_low = pce_med - cohort_table["pce_q25"].to_numpy()
    pce_high = cohort_table["pce_q75"].to_numpy() - pce_med
    axes[0, 1].bar(x, pce_med, color=colors, edgecolor="#222222", linewidth=0.8)
    axes[0, 1].errorbar(
        x, pce_med, yerr=np.vstack([pce_low, pce_high]), fmt="none", color="#222222"
    )
    axes[0, 1].set_xticks(x, [labels[v] for v in cohort_table["cohort"]])
    axes[0, 1].set_ylabel("PCE (%)")
    axes[0, 1].set_title("B  Target distribution, median and IQR")

    order = [
        "01_original_database",
        "02_after_google_drive",
        "03_after_google_drive_plus_expansion",
        "web_added_only",
        "accepted_top_half_completeness",
        "accepted_all_weight_0.5",
    ]
    for axis, split, title in [
        (axes[1, 0], "doi_grouped", "C  Fixed DOI-disjoint holdout"),
        (axes[1, 1], "row_random", "D  Fixed row-random holdout"),
    ]:
        subset = model_table.loc[model_table["split_strategy"].eq(split)].set_index("variant")
        values = [float(subset.loc[item, "r2"]) for item in order]
        bar_colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E69F00", "#59A14F", "#9C755F"]
        axis.bar(
            np.arange(len(order)), values, color=bar_colors, edgecolor="#222222", linewidth=0.8
        )
        axis.axhline(values[0], color="#333333", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(order)), [labels[item] for item in order], fontsize=8)
        axis.set_ylabel("R²")
        axis.set_title(title)
        floor = min(0.0, min(values) - 0.05)
        axis.set_ylim(floor, max(values) + 0.08)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Literature expansion: structured-data enrichment and fixed-holdout PCE utility",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "No rows."
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
        else:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
        formatted[column] = formatted[column].str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    path: Path,
    cohort_table: pd.DataFrame,
    eligibility: pd.DataFrame,
    model_table: pd.DataFrame,
    deltas: pd.DataFrame,
    high_threshold: float,
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
) -> None:
    original = cohort_table.set_index("cohort").loc["original"]
    added = cohort_table.set_index("cohort").loc["all_added"]
    lines = [
        "# PCE Expansion Utility Audit",
        "",
        "## Question",
        "",
        "Does LiteratureAgent expansion add more complete structured records, and do those records improve prediction on unchanged original-database holdouts?",
        "",
        "## Enrichment premise",
        "",
        f"- Original model-eligible rows have median predictor completeness {original['feature_completeness_median']:.3f} across the fixed 278-feature space.",
        f"- Added model-eligible rows have median predictor completeness {added['feature_completeness_median']:.3f}.",
        f"- The high-completeness ablation retains accepted additions at or above {high_threshold:.3f} completeness.",
        "",
        "## Raw eligibility",
        "",
        markdown_table(eligibility),
        "",
        "## Model-eligible cohort summary",
        "",
        markdown_table(cohort_table),
        "",
        "## Fixed-holdout model results",
        "",
        markdown_table(model_table),
        "",
        "## Paired DOI-cluster bootstrap changes versus original model",
        "",
        markdown_table(deltas, digits=4),
        "",
        "## Largest numeric shifts",
        "",
        markdown_table(numeric_shift.head(15))
        if not numeric_shift.empty
        else "No numeric feature met the comparison thresholds.",
        "",
        "## Largest categorical shifts",
        "",
        markdown_table(categorical_shift.head(15))
        if not categorical_shift.empty
        else "No categorical feature met the comparison thresholds.",
        "",
        "## Interpretation boundary",
        "",
        "- Better field coverage is an extraction result, not automatic evidence of better predictive utility.",
        "- Fixed original-distribution holdouts test backward compatibility with the original database population.",
        "- They do not test performance on a future population drawn from the expanded literature distribution.",
        "- A performance decline therefore identifies a data-integration or modeling problem to resolve; it does not erase the added records or their provenance value.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frozen = load_module(
        ROOT / "scripts" / "compare_pce_frozen_production_extratrees.py",
        "frozen_comparison_for_utility_audit",
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
        args.original_csv,
        model,
        cfg,
        args.out_dir / "preparation_audit" / "original",
    )
    integrated, integrated_target, integrated_features, _, integrated_group = frozen.prepare(
        args.integrated_csv,
        model,
        cfg,
        args.out_dir / "preparation_audit" / "integrated",
        feature_cols=features,
    )
    curated, _, _, _, _ = frozen.prepare(
        args.google_drive_csv,
        model,
        cfg,
        args.out_dir / "preparation_audit" / "curated",
        feature_cols=features,
    )
    if target != integrated_target or group_col != integrated_group:
        raise RuntimeError("Prepared datasets do not share target/group definitions.")
    if features != integrated_features:
        raise RuntimeError("Integrated preparation did not preserve the fixed feature space.")

    identity_cols = ["Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]
    curated_keys, _ = frozen.build_row_keys(curated, group_col, identity_cols)
    integrated_keys, _ = frozen.build_row_keys(integrated, group_col, identity_cols)
    curated_key_set = set(curated_keys.astype("string"))
    source = truthy(integrated["_source_added_by_literature_agent"])
    status = normalized_text(integrated["_lit_agent_confidence_status"])
    accepted_source = source & status.isin(["accepted", "sparse_accepted"])
    curated_match = integrated_keys.astype("string").isin(curated_key_set)
    cohort = pd.Series("original", index=integrated.index, dtype="string")
    cohort.loc[accepted_source & curated_match] = "curated_added"
    cohort.loc[accepted_source & ~curated_match] = "web_added"
    completeness = feature_completeness(integrated, features)

    raw = model.clean_df(model.robust_read_csv(args.integrated_csv))
    eligibility = raw_eligibility_summary(raw, model, cfg, target)
    cohort_table = cohort_summary(
        integrated, cohort, completeness, target=target, group_col=group_col
    )
    fill_shift, numeric_shift, categorical_shift = distribution_shift_tables(
        integrated, cohort, features, model
    )

    ablation_metrics, deltas, high_threshold = run_fixed_holdout_ablations(
        original=original,
        integrated=integrated,
        cohort=cohort,
        completeness=completeness,
        features=features,
        target=target,
        group_col=group_col,
        frozen=frozen,
        model=model,
        cfg=cfg,
        frozen_dir=args.frozen_dir,
        out_dir=args.out_dir,
        iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    existing = pd.read_csv(args.frozen_dir / "frozen_production_pce_comparison.csv")
    existing = existing.rename(columns={"dataset_version": "variant"})[
        [
            "split_strategy",
            "variant",
            "train_rows",
            "frozen_test_rows",
            "r2",
            "rmse",
            "mae",
        ]
    ]
    existing = existing.rename(columns={"train_rows": "total_train_rows"})
    ablation_metrics["total_train_rows"] = (
        ablation_metrics["original_train_rows"] + ablation_metrics["added_train_rows"]
    )
    model_table = pd.concat(
        [
            existing,
            ablation_metrics[
                [
                    "split_strategy",
                    "variant",
                    "total_train_rows",
                    "frozen_test_rows",
                    "r2",
                    "rmse",
                    "mae",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    eligibility.to_csv(args.out_dir / "raw_pce_eligibility_by_status.csv", index=False)
    cohort_table.to_csv(args.out_dir / "model_eligible_cohort_summary.csv", index=False)
    fill_shift.to_csv(args.out_dir / "feature_fill_rate_shift.csv", index=False)
    numeric_shift.to_csv(args.out_dir / "numeric_feature_distribution_shift.csv", index=False)
    categorical_shift.to_csv(
        args.out_dir / "categorical_feature_distribution_shift.csv", index=False
    )
    model_table.to_csv(args.out_dir / "fixed_holdout_ablation_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "fixed_holdout_ablation_paired_deltas.csv", index=False)

    make_figure(
        cohort_table,
        model_table,
        args.out_dir / "figure_pce_expansion_utility_audit.png",
        args.out_dir / "figure_pce_expansion_utility_audit.pdf",
    )
    write_report(
        args.out_dir / "PCE_EXPANSION_UTILITY_AUDIT.md",
        cohort_table,
        eligibility,
        model_table,
        deltas,
        high_threshold,
        numeric_shift,
        categorical_shift,
    )
    manifest = {
        "original_csv": str(args.original_csv.resolve()),
        "google_drive_csv": str(args.google_drive_csv.resolve()),
        "integrated_csv": str(args.integrated_csv.resolve()),
        "model_script": str(args.model_script.resolve()),
        "feature_count": len(features),
        "target": target,
        "group_column": group_col,
        "n_estimators": args.n_estimators,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_state": args.random_state,
        "curated_model_eligible_key_matches": int(
            (accepted_source & curated_match).sum()
        ),
        "web_model_eligible_rows": int((accepted_source & ~curated_match).sum()),
        "high_completeness_threshold": high_threshold,
    }
    (args.out_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(model_table.to_string(index=False))


if __name__ == "__main__":
    main()
