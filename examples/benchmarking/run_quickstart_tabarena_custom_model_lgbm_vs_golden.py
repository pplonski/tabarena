from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def _import_golden_features():
    try:
        from features_goldmine import GoldenFeatures

        return GoldenFeatures
    except ImportError:
        sandbox_dir = Path(__file__).resolve().parents[3]
        candidate_repo = sandbox_dir / "features_goldmine"
        if candidate_repo.exists():
            sys.path.insert(0, str(candidate_repo))
            from features_goldmine import GoldenFeatures

            return GoldenFeatures
        raise


GoldenFeatures = _import_golden_features()


def _cast_categoricals(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return X, []
    X_out = X.copy()
    for col in cat_cols:
        X_out[col] = X_out[col].astype("category")
    return X_out, cat_cols


class SimpleLightGBM(AbstractExecModel):
    def __init__(self, hyperparameters: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.hyperparameters = hyperparameters or {}

    def get_model_cls(self):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if self.problem_type in ["binary", "multiclass"]:
            return LGBMClassifier
        if self.problem_type == "regression":
            return LGBMRegressor
        raise AssertionError(f"LightGBM does not recognize problem_type='{self.problem_type}'")

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        model_cls = self.get_model_cls()
        X_fit, cat_cols = _cast_categoricals(X)
        self._cat_cols = cat_cols
        self.model = model_cls(**self.hyperparameters)
        self.model.fit(X_fit, y, categorical_feature=cat_cols if cat_cols else "auto")
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        X_in, _ = _cast_categoricals(X)
        y_pred = self.model.predict(X_in)
        return pd.Series(y_pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError("_predict_proba is only valid for classification.")
        X_in, _ = _cast_categoricals(X)
        y_pred_proba = self.model.predict_proba(X_in)
        return pd.DataFrame(y_pred_proba, columns=self.model.classes_, index=X.index)


class GoldenLightGBM(SimpleLightGBM):
    def __init__(self, hyperparameters: dict | None = None, golden_kwargs: dict | None = None, **kwargs):
        super().__init__(hyperparameters=hyperparameters, **kwargs)
        self.golden_kwargs = golden_kwargs or {}

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.golden_kwargs["selectivity"] = "strict"
        self.gf = GoldenFeatures(**self.golden_kwargs)
        X_gold = self.gf.fit_transform(X, y)
        print(
            f"[GoldenFeatures] selected={len(self.gf.golden_features_)} features (strict): "
            f"{[g.name for g in self.gf.golden_features_]}",
            flush=True,
        )
        X_aug = pd.concat([X, X_gold], axis=1)
        return super()._fit(X_aug, y, **kwargs)

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        X_gold = self.gf.transform(X)
        X_aug = pd.concat([X, X_gold], axis=1)
        return super()._predict(X_aug)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X_gold = self.gf.transform(X)
        X_aug = pd.concat([X, X_gold], axis=1)
        return super()._predict_proba(X_aug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline LGBM vs LGBM+GoldenFeatures on TabArena datasets")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--problem-type", choices=["all", "regression", "binary", "multiclass"], default="all")
    parser.add_argument("--max-train-samples", type=int, default=1000000)
    parser.add_argument("--max-datasets", type=int, default=0, help="0 means all selected datasets")
    parser.add_argument("--selectivity", choices=["strict"], default="strict")
    parser.add_argument("--golden-verbose", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    root = Path(__file__).parent
    expname = str(root / "experiments" / "quickstart_lgbm_vs_golden")
    leaderboard_dir = str(root / "leaderboards" / "quickstart_lgbm_vs_golden")

    ta_context = TabArenaContext()
    task_metadata = ta_context.task_metadata.copy()

    selected = task_metadata.copy()
    if args.problem_type != "all":
        selected = selected[selected["problem_type"] == args.problem_type].copy()
    selected = selected[selected["n_samples_train_per_fold"] < args.max_train_samples].copy()
    datasets = sorted(selected["name"].tolist())
    if args.max_datasets > 0:
        datasets = datasets[: args.max_datasets]

    print(
        f"Selected datasets={len(datasets)} | problem_type={args.problem_type} "
        f"| max_train_samples={args.max_train_samples} | fold={args.fold}"
    )
    if len(datasets) <= 30:
        print(datasets)

    methods = [
        Experiment(
            name="LGBM_Baseline",
            method_cls=SimpleLightGBM,
            method_kwargs={
                "hyperparameters": {
                    "n_estimators": args.n_estimators,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "n_jobs": -1,
                    "verbosity": -1,
                    "random_state": 0,
                }
            },
        ),
        Experiment(
            name="LGBM_GoldenFeatures",
            method_cls=GoldenLightGBM,
            method_kwargs={
                "hyperparameters": {
                    "n_estimators": args.n_estimators,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "n_jobs": -1,
                    "verbosity": -1,
                    "random_state": 0,
                },
                "golden_kwargs": {
                    "random_state": 0,
                    "verbose": args.golden_verbose,
                    "selectivity": args.selectivity,
                },
            },
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)
    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=[args.fold],
        methods=methods,
        ignore_cache=args.ignore_cache,
    )

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1200):
        print("\nModel results (first 200 rows):")
        print(end_to_end_results.model_results.head(200))

    leaderboard = end_to_end_results.compare_on_tabarena(
        output_dir=leaderboard_dir,
        only_valid_tasks=True,
        plot_with_baselines=True,
    )
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("\nLeaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print(f"\nSaved figures/tables to: {leaderboard_dir}")


if __name__ == "__main__":
    main()
