import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pce_then_stability_same_approach.py"
)


def load_model_module():
    spec = importlib.util.spec_from_file_location("pce_model_test_module", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extra_trees_honors_requested_estimator_count():
    module = load_model_module()
    cfg = module.Config(MODEL_BACKEND="extra_trees", N_ESTIMATORS=650)
    model = module.make_model("regression", cfg)
    metadata = module.estimator_metadata(model)
    assert metadata["model_backend"] == "extra_trees"
    assert metadata["actual_n_estimators"] == 650


def test_grouped_cv_keeps_each_group_in_one_side_of_fold():
    module = load_model_module()
    y = pd.Series(np.arange(20, dtype=float))
    groups = pd.Series(np.repeat(["a", "b", "c", "d", "e"], 4))
    cfg = module.Config(CV_STRATEGY="grouped", N_SPLITS=5)
    folds = list(module.make_cv(y, groups, cfg))
    for train_index, test_index in folds:
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        assert train_groups.isdisjoint(test_groups)


def test_row_random_cv_is_explicit_and_reproducible():
    module = load_model_module()
    y = pd.Series(np.arange(30, dtype=float))
    groups = pd.Series(np.repeat(["a", "b", "c", "d", "e"], 6))
    cfg = module.Config(CV_STRATEGY="row_random", N_SPLITS=5, RANDOM_STATE=42)
    first = [(train.tolist(), test.tolist()) for train, test in module.make_cv(y, groups, cfg)]
    second = [(train.tolist(), test.tolist()) for train, test in module.make_cv(y, groups, cfg)]
    assert first == second
    assert any(
        set(groups.iloc[train]) & set(groups.iloc[test])
        for train, test in first
    )


if __name__ == "__main__":
    test_extra_trees_honors_requested_estimator_count()
    test_grouped_cv_keeps_each_group_in_one_side_of_fold()
    test_row_random_cv_is_explicit_and_reproducible()
    print("PCE model configuration tests passed")
