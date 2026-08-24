from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pce_then_stability_same_approach.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("pce_stability_model_test", MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_cv_keeps_doi_groups_disjoint():
    module = load_model_module()
    y = pd.Series(np.arange(30, dtype=float))
    groups = pd.Series([f"doi_{i // 3}" for i in range(30)])

    splits = list(module.make_cv(y, groups, module.Config()))

    assert len(splits) == module.Config().N_SPLITS
    for train_idx, test_idx in splits:
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))


def test_hierarchical_residual_pce_writes_grouped_oof_outputs(tmp_path):
    module = load_model_module()
    rng = np.random.default_rng(7)
    n_groups = 20
    rows_per_group = 3
    n = n_groups * rows_per_group
    chemistry = rng.normal(size=n)
    process = rng.normal(size=n)
    interface = rng.normal(size=n)
    context = rng.normal(size=n)
    df = pd.DataFrame({
        "Ref_DOI_number": np.repeat([f"10.1/test.{i}" for i in range(n_groups)], rows_per_group),
        "chem_feature": chemistry,
        "process_feature": process,
        "interface_feature": interface,
        "context_feature": context,
        "JV_default_PCE": 12 + 1.5 * chemistry + process + 0.7 * interface + 0.4 * context,
    })
    blocks = {
        "chemistry_architecture": ["chem_feature"],
        "process": ["process_feature"],
        "interfaces": ["interface_feature"],
        "device_context": ["context_feature"],
        "other": [],
    }
    cfg = module.Config()
    cfg.MIN_ROWS_REGRESSION = 20
    cfg.N_SPLITS = 5
    original_make_model = module.make_model
    module.make_model = lambda task, _cfg: ExtraTreesRegressor(
        n_estimators=20, min_samples_leaf=2, random_state=7, n_jobs=1
    )
    try:
        predictions, metrics, stages = module.train_hierarchical_residual_pce(
            df,
            blocks,
            "JV_default_PCE",
            "Ref_DOI_number",
            cfg,
            tmp_path,
        )
    finally:
        module.make_model = original_make_model

    assert metrics["status"] == "trained"
    assert metrics["n_stages"] == 4
    assert len(stages) == 4
    assert predictions["oof_pred"].notna().all()
    assert (tmp_path / "pce_hierarchical_residual_metrics.json").exists()
    assert (tmp_path / "pce_hierarchical_residual_stage_metrics.csv").exists()
