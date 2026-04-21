from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.nips2025_utils.end_to_end import EndToEnd
from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def print_dataset_statistics(task_metadata_subset: pd.DataFrame) -> None:
    rows = []
    for _, row in task_metadata_subset.iterrows():
        n_features = int(row.get("n_features", np.nan))
        n_num = int(row.get("NumberOfNumericFeatures", 0) or 0)
        n_cat = int(row.get("NumberOfSymbolicFeatures", 0) or 0)
        n_other = max(0, n_features - n_num - n_cat)

        n_missing = int(row.get("NumberOfMissingValues", 0) or 0)
        n_rows_w_missing = int(row.get("NumberOfInstancesWithMissingValues", 0) or 0)
        n_instances = int(row.get("NumberOfInstances", 0) or 0)
        missing_cell_pct = (100.0 * n_missing / max(1, n_instances * max(1, n_features)))
        missing_row_pct = (100.0 * n_rows_w_missing / max(1, n_instances))

        rows.append(
            {
                "dataset": row["name"],
                "problem_type": row["problem_type"],
                "n_instances": n_instances,
                "n_train_fold": int(row["n_samples_train_per_fold"]),
                "n_test_fold": int(row["n_samples_test_per_fold"]),
                "n_features": n_features,
                "n_numeric": n_num,
                "n_categorical": n_cat,
                "n_other": n_other,
                "n_classes": int(row.get("n_classes", 0) or 0),
                "missing_cells": n_missing,
                "missing_rows": n_rows_w_missing,
                "missing_cell_pct": missing_cell_pct,
                "missing_row_pct": missing_row_pct,
            }
        )

    df_stats = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

    print("\n=== Dataset Summary Before Fit ===")
    print(f"n_datasets={len(df_stats)}")
    print(
        "train_samples_per_fold: "
        f"min={df_stats['n_train_fold'].min()}, "
        f"median={df_stats['n_train_fold'].median():.0f}, "
        f"max={df_stats['n_train_fold'].max()}"
    )
    print(
        "n_features: "
        f"min={df_stats['n_features'].min()}, "
        f"median={df_stats['n_features'].median():.0f}, "
        f"max={df_stats['n_features'].max()}"
    )
    print(
        "missing_cells_total="
        f"{int(df_stats['missing_cells'].sum())}, "
        f"missing_rows_total={int(df_stats['missing_rows'].sum())}"
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df_stats.to_string(index=False))


def export_openml_split_to_csv(
    *,
    task_id: int,
    dataset_name: str,
    output_dir: Path,
    fold: int = 0,
    repeat: int = 0,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold, repeat=repeat)

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True).rename(task.label)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename(task.label)], axis=1)

    train_path = output_dir / f"{dataset_name}_fold{fold}_repeat{repeat}_train.csv"
    test_path = output_dir / f"{dataset_name}_fold{fold}_repeat{repeat}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def log_dataset_categorical_details(*, task_id: int, dataset_name: str, max_cols: int = 200) -> None:
    task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
    X = task.X
    cat_mask = X.dtypes.astype(str).isin(["object", "category", "bool"])
    cat_cols = X.columns[cat_mask].tolist()

    print(f"\n=== Categorical Details: {dataset_name} ===")
    print(f"detected_categorical_columns={len(cat_cols)} out of total_columns={X.shape[1]}")
    if not cat_cols:
        print("No categorical columns detected by dtype (object/category/bool).")
        return

    stats = []
    for col in cat_cols[:max_cols]:
        s = X[col]
        stats.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "n_missing": int(s.isna().sum()),
                "missing_pct": float(100.0 * s.isna().mean()),
            }
        )
    df_cat = pd.DataFrame(stats)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df_cat.to_string(index=False))
    if len(cat_cols) > max_cols:
        print(f"... truncated: showing first {max_cols}/{len(cat_cols)} categorical columns")


class OptunaCVLightGBM(AbstractExecModel):
    def __init__(
        self,
        n_splits: int = 8,
        n_trials: int = 20,
        timeout: int | None = 600,
        random_state: int = 0,
        n_jobs: int = -1,
        fixed_params: dict | None = None,
        optuna_verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fixed_params = fixed_params or {}
        self.optuna_verbose = optuna_verbose

        self.fold_models = []
        self.fold_categorical_features = []
        self.best_params = None
        self.classes_ = None

    def get_model_cls(self):
        from lightgbm import LGBMClassifier, LGBMRegressor

        is_classification = self.problem_type in ["binary", "multiclass"]
        if is_classification:
            return LGBMClassifier
        if self.problem_type == "regression":
            return LGBMRegressor
        raise AssertionError(f"LightGBM does not recognize problem_type='{self.problem_type}'")

    def _make_cv(self, y: pd.Series):
        if self.problem_type in ["binary", "multiclass"]:
            class_counts = y.value_counts()
            min_class_count = int(class_counts.min())
            n_splits = min(self.n_splits, min_class_count)
            if n_splits < 2:
                raise AssertionError(
                    f"Need at least 2 samples per class for CV, got min_class_count={min_class_count}"
                )
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            return cv, n_splits
        n_splits = min(self.n_splits, len(y))
        if n_splits < 2:
            raise AssertionError(f"Need at least 2 samples for CV, got n_samples={len(y)}")
        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return cv, n_splits

    def _sample_params(self, trial: optuna.Trial) -> dict:
        # Conservative and robust search space for tabular LightGBM.
        params = {
            "n_estimators": 1000, # trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 5e-4, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 2550),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            #"min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
        }
        params.update(self.fixed_params)
        return params

    @staticmethod
    def _trees_used(model) -> int | None:
        # Prefer explicit early-stopping best iteration when available.
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            return int(best_iter)
        # sklearn wrapper sometimes exposes fitted estimators count instead.
        n_estimators_fit = getattr(model, "n_estimators_", None)
        if n_estimators_fit is not None:
            return int(n_estimators_fit)
        # Booster fallback.
        booster = getattr(model, "booster_", None)
        if booster is not None:
            cur_iter = booster.current_iteration()
            if cur_iter is not None:
                return int(cur_iter)
        return None

    def _metric_error_from_val(self, y_val: pd.Series, model, X_val: pd.DataFrame) -> float:
        if self.problem_type == "regression":
            y_pred = model.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, y_pred)))

        y_proba = model.predict_proba(X_val)
        if self.problem_type == "binary":
            # Minimize metric_error, so optimize 1 - AUC.
            try:
                return float(1.0 - roc_auc_score(y_val, y_proba[:, 1]))
            except ValueError:
                # Fallback for degenerate folds.
                return float(log_loss(y_val, y_proba, labels=model.classes_))
        return float(log_loss(y_val, y_proba, labels=model.classes_))

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from lightgbm import early_stopping

        optuna.logging.set_verbosity(optuna.logging.INFO if self.optuna_verbose else optuna.logging.WARNING)

        model_cls = self.get_model_cls()
        cv, n_splits_actual = self._make_cv(y=y)
        split_indices = list(cv.split(X, y if self.problem_type in ["binary", "multiclass"] else None))

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial=trial)
            fold_errors = []
            fold_best_iters = []
            for train_idx, val_idx in split_indices:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
                if trial.number == 0 and len(fold_errors) == 0:
                    print(f"[Categoricals][CV] detected columns: {cat_cols}", flush=True)
                if cat_cols:
                    X_train = X_train.copy()
                    X_val = X_val.copy()
                    for col in cat_cols:
                        X_train[col] = X_train[col].astype("category")
                        X_val[col] = X_val[col].astype("category")

                model = model_cls(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=cat_cols if cat_cols else "auto",
                    callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
                )
                fold_errors.append(self._metric_error_from_val(y_val=y_val, model=model, X_val=X_val))
                trees_used = self._trees_used(model=model)
                if trees_used is not None:
                    fold_best_iters.append(trees_used)
            if fold_best_iters:
                trial.set_user_attr("best_iter_min", int(np.min(fold_best_iters)))
                trial.set_user_attr("best_iter_median", int(np.median(fold_best_iters)))
                trial.set_user_attr("best_iter_max", int(np.max(fold_best_iters)))
            return float(np.mean(fold_errors))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        def trial_callback(study_obj: optuna.Study, frozen_trial: optuna.trial.FrozenTrial):
            if self.optuna_verbose:
                trial_value = frozen_trial.value
                best_value = study_obj.best_value if len(study_obj.trials) > 0 else None
                bi_min = frozen_trial.user_attrs.get("best_iter_min", "n/a")
                bi_med = frozen_trial.user_attrs.get("best_iter_median", "n/a")
                bi_max = frozen_trial.user_attrs.get("best_iter_max", "n/a")
                print(
                    f"[Optuna] Trial {frozen_trial.number}: "
                    f"value={trial_value if trial_value is not None else 'n/a'}, "
                    f"best={best_value if best_value is not None else 'n/a'}, "
                    f"best_iter(min/med/max)=({bi_min}/{bi_med}/{bi_max})",
                    flush=True,
                )

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[trial_callback])

        self.best_params = self._sample_params(trial=study.best_trial)
        self.fold_models = []
        self.fold_categorical_features = []
        fold_tree_stats = []
        for train_idx, val_idx in split_indices:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
            if cat_cols:
                X_train = X_train.copy()
                X_val = X_val.copy()
                for col in cat_cols:
                    X_train[col] = X_train[col].astype("category")
                    X_val[col] = X_val[col].astype("category")
            model = model_cls(**self.best_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                categorical_feature=cat_cols if cat_cols else "auto",
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
            )
            self.fold_models.append(model)
            self.fold_categorical_features.append(cat_cols)
            best_iter = self._trees_used(model=model)
            print(
                f"[FinalCV] fold={len(self.fold_models)-1} trees_used={best_iter} "
                f"configured_n_estimators={getattr(model, 'n_estimators', None)}",
                flush=True,
            )
            fold_tree_stats.append(
                {
                    "fold": len(self.fold_models) - 1,
                    "best_iteration_": int(best_iter) if best_iter is not None else None,
                    "n_estimators_configured": int(getattr(model, "n_estimators", -1)),
                }
            )

        if self.problem_type in ["binary", "multiclass"]:
            self.classes_ = self.fold_models[0].classes_

        print(
            f"Optuna finished for {self.problem_type}: "
            f"best_value={study.best_value:.6f}, n_splits={n_splits_actual}, n_trials={len(study.trials)}"
        )
        print(f"Best params: {self.best_params}")
        best_iters = [d["best_iteration_"] for d in fold_tree_stats if d["best_iteration_"] is not None]
        if best_iters:
            print(
                "Early stopping tree summary: "
                f"min={min(best_iters)}, median={int(np.median(best_iters))}, max={max(best_iters)}"
            )
        print("Per-fold tree stats:\n" + pformat(fold_tree_stats))
        return self

    @staticmethod
    def _predict_with_best_iteration(model, X: pd.DataFrame) -> np.ndarray:
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            return model.predict(X, num_iteration=best_iter)
        return model.predict(X)

    @staticmethod
    def _predict_proba_with_best_iteration(model, X: pd.DataFrame) -> np.ndarray:
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            return model.predict_proba(X, num_iteration=best_iter)
        return model.predict_proba(X)

    @staticmethod
    def _cast_categoricals_for_inference(X: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
        if not categorical_features:
            return X
        X_out = X.copy()
        for col in categorical_features:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype("category")
        return X_out

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self.problem_type == "regression":
            preds = np.column_stack(
                [
                    self._predict_with_best_iteration(
                        m,
                        self._cast_categoricals_for_inference(X, cat_cols),
                    )
                    for m, cat_cols in zip(self.fold_models, self.fold_categorical_features)
                ]
            )
            y_pred = preds.mean(axis=1)
            return pd.Series(y_pred, index=X.index)

        proba_df = self._predict_proba(X)
        pred_idx = np.argmax(proba_df.to_numpy(), axis=1)
        y_pred = pd.Series(self.classes_[pred_idx], index=X.index)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError("_predict_proba is only valid for classification.")
        proba = np.stack(
            [
                self._predict_proba_with_best_iteration(
                    m,
                    self._cast_categoricals_for_inference(X, cat_cols),
                )
                for m, cat_cols in zip(self.fold_models, self.fold_categorical_features)
            ],
            axis=0,
        ).mean(axis=0)
        return pd.DataFrame(proba, columns=self.classes_, index=X.index)


if __name__ == "__main__":
    expname = str(Path(__file__).parent / "experiments" / "quickstart_custom_model_lgbm_optuna_cv8")
    leaderboard_dir = str(Path(__file__).parent / "leaderboards" / "quickstart_custom_model_lgbm_optuna_cv8")
    ignore_cache = True

    ta_context = TabArenaContext()
    task_metadata = ta_context.task_metadata.copy()

    # Run only regression tasks in the "small" bucket:
    # n_samples_train_per_fold < 10_000
    task_metadata_small_regression = task_metadata[
        (task_metadata["problem_type"] == "regression")
        & (task_metadata["n_samples_train_per_fold"] < 10_000)
    ].copy()
    datasets = ["wine_quality"]
    task_metadata_selected = task_metadata_small_regression[
        task_metadata_small_regression["name"].isin(datasets)
    ].copy()
    print(f"Selected {len(datasets)} dataset: {datasets}")
    print_dataset_statistics(task_metadata_subset=task_metadata_selected)

    # Export the benchmark split used in this script (TabArena-Lite style split).
    export_dir = Path(__file__).parent / "data_exports" / "wine_quality"
    selected_row = task_metadata_selected[task_metadata_selected["name"] == "wine_quality"].iloc[0]
    log_dataset_categorical_details(
        task_id=int(selected_row["tid"]),
        dataset_name="wine_quality",
    )
    train_csv, test_csv = export_openml_split_to_csv(
        task_id=int(selected_row["tid"]),
        dataset_name="wine_quality",
        output_dir=export_dir,
        fold=0,
        repeat=0,
    )
    print(f"Saved train CSV: {train_csv}")
    print(f"Saved test CSV: {test_csv}")

    folds = [0]  # TabArena-Lite style

    methods = [
        Experiment(
            name="MyCustomModel_LGBM_OptunaCV8",
            method_cls=OptunaCVLightGBM,
            method_kwargs={
                "n_splits": 8,
                "n_trials": 200,
                "timeout": 3600,
                "random_state": 0,
                "n_jobs": -1,
            },
        ),
    ]

    exp_batch_runner = ExperimentBatchRunner(expname=expname, task_metadata=task_metadata)

    results_lst: list[dict[str, Any]] = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=methods,
        ignore_cache=ignore_cache,
    )

    end_to_end = EndToEnd.from_raw(results_lst=results_lst, task_metadata=task_metadata, cache=False, cache_raw=False)
    end_to_end_results = end_to_end.to_results()

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
        print(f"Results:\n{end_to_end_results.model_results.head(100)}")

    leaderboard: pd.DataFrame = end_to_end_results.compare_on_tabarena(
        output_dir=leaderboard_dir,
        only_valid_tasks=True,
        plot_with_baselines=True,
    )
    leaderboard_website = ta_context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("Leaderboard:")
    print(leaderboard_website.to_markdown(index=False))
    print(f"View saved figures in {leaderboard_dir}")
