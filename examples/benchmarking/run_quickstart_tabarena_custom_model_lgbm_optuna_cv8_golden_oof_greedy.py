from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import sys

from tabarena.benchmark.experiment import Experiment, ExperimentBatchRunner
from tabarena.benchmark.models.wrapper.abstract_class import AbstractExecModel
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
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
        max_greedy_ensemble_size: int = 50,
        greedy_tol: float = 1e-7,
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
        self.max_greedy_ensemble_size = max_greedy_ensemble_size
        self.greedy_tol = greedy_tol
        self.fixed_params = fixed_params or {}
        self.optuna_verbose = optuna_verbose

        self.fold_models = []
        self.fold_categorical_features = []
        self.fold_golden_features = []
        self.use_golden_features_ = True
        self.best_params = None
        self.classes_ = None
        self.positive_class_ = None
        self.trial_records: list[dict[str, Any]] = []
        self.greedy_selected_trial_indices: list[int] = []
        self.greedy_trial_weights: dict[int, float] = {}
        self.greedy_history: list[dict[str, Any]] = []

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
        use_regularization = trial.suggest_categorical("use_regularization", [False, True])
        # Conservative and robust search space for tabular LightGBM.
        params = {
            "n_estimators": 5000, # trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2**10-1, log=True),
            "max_depth": -1, #trial.suggest_int("max_depth", 3, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "bagging_freq": 1, #trial.suggest_int("bagging_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            #"min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": -1,
        }
        if use_regularization:
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True)
        else:
            params["reg_alpha"] = 0.0
            params["reg_lambda"] = 0.0
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

    def _metric_error_from_predictions(self, y_true: pd.Series, y_pred_oof: np.ndarray) -> float:
        if self.problem_type == "regression":
            return float(np.sqrt(mean_squared_error(y_true, y_pred_oof)))
        if self.problem_type == "binary":
            return float(1.0 - roc_auc_score(y_true, y_pred_oof))
        return float(log_loss(y_true, y_pred_oof, labels=self.classes_))

    def _empty_oof_buffer(self, n_rows: int) -> np.ndarray:
        if self.problem_type == "regression":
            return np.zeros(n_rows, dtype=np.float64)
        if self.problem_type == "binary":
            return np.zeros(n_rows, dtype=np.float64)
        return np.zeros((n_rows, len(self.classes_)), dtype=np.float64)

    def _align_multiclass_proba(self, proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
        out = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        cls_to_col = {c: i for i, c in enumerate(self.classes_)}
        for j, cls in enumerate(model_classes):
            out[:, cls_to_col[cls]] = proba[:, j]
        return out

    def _extract_binary_proba(self, model, X_val: pd.DataFrame) -> np.ndarray:
        proba = self._predict_proba_with_best_iteration(model, X_val)
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            return proba[:, 1]
        cls_to_col = {c: i for i, c in enumerate(model_classes)}
        pos_col = cls_to_col[self.positive_class_]
        return proba[:, pos_col]

    def _predict_val_for_oof(self, model, X_val: pd.DataFrame) -> np.ndarray:
        if self.problem_type == "regression":
            return self._predict_with_best_iteration(model, X_val)
        if self.problem_type == "binary":
            return self._extract_binary_proba(model, X_val)
        proba = self._predict_proba_with_best_iteration(model, X_val)
        return self._align_multiclass_proba(proba=proba, model_classes=model.classes_)

    def _greedy_select_trials(self, y_true: pd.Series) -> tuple[list[int], dict[int, float], list[dict[str, Any]]]:
        trial_preds = [t["oof_pred"] for t in self.trial_records]
        if not trial_preds:
            raise AssertionError("No successful trials available for greedy ensemble.")

        selected: list[int] = []
        history: list[dict[str, Any]] = []
        current_score = np.inf
        ensemble_sum: np.ndarray | None = None
        max_steps = min(self.max_greedy_ensemble_size, max(1, len(trial_preds) * 2))

        for step in range(max_steps):
            best_idx = None
            best_score = current_score
            best_candidate_pred = None
            n_selected = len(selected)
            for trial_idx, pred in enumerate(trial_preds):
                candidate_pred = pred if n_selected == 0 else (ensemble_sum + pred) / (n_selected + 1)
                score = self._metric_error_from_predictions(y_true=y_true, y_pred_oof=candidate_pred)
                if score < best_score - self.greedy_tol:
                    best_score = score
                    best_idx = trial_idx
                    best_candidate_pred = candidate_pred

            if best_idx is None:
                break

            selected.append(best_idx)
            ensemble_sum = trial_preds[best_idx].copy() if n_selected == 0 else (ensemble_sum + trial_preds[best_idx])
            improvement = current_score - best_score if np.isfinite(current_score) else np.nan
            history.append(
                {
                    "step": step,
                    "trial_idx": best_idx,
                    "trial_number": self.trial_records[best_idx]["trial_number"],
                    "score": float(best_score),
                    "improvement": float(improvement) if np.isfinite(improvement) else np.nan,
                }
            )
            current_score = best_score

        counts: dict[int, int] = {}
        for idx in selected:
            counts[idx] = counts.get(idx, 0) + 1
        total = max(1, len(selected))
        weights = {idx: c / total for idx, c in counts.items()}
        return selected, weights, history

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from lightgbm import early_stopping

        optuna.logging.set_verbosity(optuna.logging.INFO if self.optuna_verbose else optuna.logging.WARNING)

        y = y.copy()
        if self.problem_type in ["binary", "multiclass"]:
            self.classes_ = np.sort(pd.unique(y))
            if self.problem_type == "binary":
                if len(self.classes_) != 2:
                    raise AssertionError(f"Expected exactly 2 classes for binary, got {self.classes_}")
                self.positive_class_ = self.classes_[1]

        model_cls = self.get_model_cls()
        cv, n_splits_actual = self._make_cv(y=y)
        split_indices = list(cv.split(X, y if self.problem_type in ["binary", "multiclass"] else None))
        print("[GF] Precomputing fold-safe golden features (fit on train fold only)", flush=True)

        prepared_splits = []
        self.fold_golden_features = []
        for fold_id, (train_idx, val_idx) in enumerate(split_indices):
            X_train_raw = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            gf = GoldenFeatures(random_state=self.random_state + fold_id, verbose=0, selectivity='strict')
            X_gold_train = gf.fit_transform(X_train_raw, y_train)
            X_gold_val = gf.transform(X_val_raw)

            X_train_aug = pd.concat([X_train_raw, X_gold_train], axis=1)
            X_val_aug = pd.concat([X_val_raw, X_gold_val], axis=1)
            cat_cols = X_train_raw.select_dtypes(include=["category", "object"]).columns.tolist()

            prepared_splits.append(
                {
                    "X_train_raw": X_train_raw,
                    "X_train": X_train_aug,
                    "y_train": y_train,
                    "X_val_raw": X_val_raw,
                    "X_val": X_val_aug,
                    "y_val": y_val,
                    "cat_cols": cat_cols,
                    "gf": gf,
                }
            )
            print(
                f"[GF] fold={fold_id} selected={len(gf.golden_features_)} "
                f"aug_features={X_train_aug.shape[1]}",
                flush=True,
            )

        def objective(trial: optuna.Trial) -> float:
            use_golden_features = True # trial.suggest_categorical("use_golden_features", [False, True])
            params = self._sample_params(trial=trial)
            fold_best_iters = []
            oof_pred = self._empty_oof_buffer(n_rows=len(X))
            fold_models = []
            fold_categorical_features = []
            fold_golden_features = []
            for fold_idx, split in enumerate(prepared_splits):
                X_train = split["X_train"] if use_golden_features else split["X_train_raw"]
                y_train = split["y_train"]
                X_val = split["X_val"] if use_golden_features else split["X_val_raw"]
                y_val = split["y_val"]
                cat_cols = split["cat_cols"]
                val_idx = split_indices[fold_idx][1]
                if trial.number == 0 and fold_idx == 0:
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
                val_pred = self._predict_val_for_oof(model=model, X_val=X_val)
                oof_pred[val_idx] = val_pred
                fold_models.append(model)
                fold_categorical_features.append(cat_cols)
                fold_golden_features.append(split["gf"] if use_golden_features else None)
                trees_used = self._trees_used(model=model)
                if trees_used is not None:
                    fold_best_iters.append(trees_used)
                if trial.number == 0:
                    print(
                        f"[GF][Trial0] fold={fold_idx} use_golden={use_golden_features} "
                        f"train_shape={X_train.shape} val_shape={X_val.shape}",
                        flush=True,
                    )
            metric_error = self._metric_error_from_predictions(y_true=y, y_pred_oof=oof_pred)
            if fold_best_iters:
                trial.set_user_attr("best_iter_min", int(np.min(fold_best_iters)))
                trial.set_user_attr("best_iter_median", int(np.median(fold_best_iters)))
                trial.set_user_attr("best_iter_max", int(np.max(fold_best_iters)))
            trial.set_user_attr("use_golden_features", bool(use_golden_features))
            self.trial_records.append(
                {
                    "trial_number": trial.number,
                    "params": params,
                    "use_golden_features": bool(use_golden_features),
                    "fold_models": fold_models,
                    "fold_categorical_features": fold_categorical_features,
                    "fold_golden_features": fold_golden_features,
                    "fold_best_iters": fold_best_iters,
                    "oof_pred": oof_pred,
                    "metric_error": float(metric_error),
                }
            )
            return float(metric_error)

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
                use_gf = frozen_trial.user_attrs.get("use_golden_features", "n/a")
                print(
                    f"[Optuna] Trial {frozen_trial.number}: "
                    f"value={trial_value if trial_value is not None else 'n/a'}, "
                    f"best={best_value if best_value is not None else 'n/a'}, "
                    f"use_golden={use_gf}, "
                    f"best_iter(min/med/max)=({bi_min}/{bi_med}/{bi_max})",
                    flush=True,
                )

        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[trial_callback])

        # Keep only successful trial records in trial-number order.
        self.trial_records = sorted(self.trial_records, key=lambda d: d["trial_number"])

        selected, weights, history = self._greedy_select_trials(y_true=y)
        self.greedy_selected_trial_indices = selected
        self.greedy_trial_weights = weights
        self.greedy_history = history

        # Compatibility fields with previous script behavior.
        best_idx = min(weights, key=lambda idx: self.trial_records[idx]["metric_error"])
        self.best_params = self.trial_records[best_idx]["params"]
        self.use_golden_features_ = self.trial_records[best_idx]["use_golden_features"]

        print(
            f"Optuna finished for {self.problem_type}: "
            f"best_value={study.best_value:.6f}, n_splits={n_splits_actual}, n_trials={len(study.trials)}"
        )
        print(f"Best params: {self.best_params}")
        print(f"Best trial use_golden_features: {self.use_golden_features_}")
        print(
            f"Greedy ensemble selected {len(selected)} members "
            f"from {len(self.trial_records)} successful trials."
        )
        print(f"Greedy unique-trial weights (record_idx -> weight): {self.greedy_trial_weights}")
        print("Greedy selection history:\n" + pformat(self.greedy_history))
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

    @staticmethod
    def _augment_with_golden_features(X: pd.DataFrame, gf) -> pd.DataFrame:
        if gf is None:
            return X
        X_gold = gf.transform(X)
        return pd.concat([X, X_gold], axis=1)

    def _predict_for_trial_record(self, X: pd.DataFrame, trial_record: dict) -> np.ndarray:
        fold_models = trial_record["fold_models"]
        fold_cat_cols = trial_record["fold_categorical_features"]
        fold_gfs = trial_record["fold_golden_features"]
        if self.problem_type == "regression":
            preds = np.column_stack(
                [
                    self._predict_with_best_iteration(
                        m,
                        self._cast_categoricals_for_inference(
                            self._augment_with_golden_features(X, gf),
                            cat_cols,
                        ),
                    )
                    for m, cat_cols, gf in zip(fold_models, fold_cat_cols, fold_gfs)
                ]
            )
            return preds.mean(axis=1)

        if self.problem_type == "binary":
            probs = np.column_stack(
                [
                    self._extract_binary_proba(
                        m,
                        self._cast_categoricals_for_inference(
                            self._augment_with_golden_features(X, gf),
                            cat_cols,
                        ),
                    )
                    for m, cat_cols, gf in zip(fold_models, fold_cat_cols, fold_gfs)
                ]
            )
            return probs.mean(axis=1)

        proba = np.stack(
            [
                self._align_multiclass_proba(
                    proba=self._predict_proba_with_best_iteration(
                        m,
                        self._cast_categoricals_for_inference(
                            self._augment_with_golden_features(X, gf),
                            cat_cols,
                        ),
                    ),
                    model_classes=m.classes_,
                )
                for m, cat_cols, gf in zip(fold_models, fold_cat_cols, fold_gfs)
            ],
            axis=0,
        ).mean(axis=0)
        return proba

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self.problem_type == "regression":
            if not self.greedy_trial_weights:
                raise AssertionError("Missing greedy trial weights. Ensure fit completed successfully.")
            y_pred = np.zeros(len(X), dtype=np.float64)
            for trial_idx, weight in self.greedy_trial_weights.items():
                trial_pred = self._predict_for_trial_record(X=X, trial_record=self.trial_records[trial_idx])
                y_pred += weight * trial_pred
            return pd.Series(y_pred, index=X.index)

        proba_df = self._predict_proba(X)
        pred_idx = np.argmax(proba_df.to_numpy(), axis=1)
        y_pred = pd.Series(self.classes_[pred_idx], index=X.index)
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.problem_type not in ["binary", "multiclass"]:
            raise AssertionError("_predict_proba is only valid for classification.")
        if not self.greedy_trial_weights:
            raise AssertionError("Missing greedy trial weights. Ensure fit completed successfully.")
        if self.problem_type == "binary":
            proba_pos = np.zeros(len(X), dtype=np.float64)
            for trial_idx, weight in self.greedy_trial_weights.items():
                trial_pred_pos = self._predict_for_trial_record(X=X, trial_record=self.trial_records[trial_idx])
                proba_pos += weight * trial_pred_pos
            proba = np.column_stack([1.0 - proba_pos, proba_pos])
            return pd.DataFrame(proba, columns=self.classes_, index=X.index)

        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for trial_idx, weight in self.greedy_trial_weights.items():
            trial_pred = self._predict_for_trial_record(X=X, trial_record=self.trial_records[trial_idx])
            proba += weight * trial_pred
        return pd.DataFrame(proba, columns=self.classes_, index=X.index)


if __name__ == "__main__":
    expname = str(Path(__file__).parent / "experiments" / "quickstart_custom_model_lgbm_optuna_cv8_golden_oof_greedy")
    leaderboard_dir = str(Path(__file__).parent / "leaderboards" / "quickstart_custom_model_lgbm_optuna_cv8_golden_oof_greedy")
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
            name="MyCustomModel_LGBM_OptunaCV8_Golden_OOF_Greedy",
            method_cls=OptunaCVLightGBM,
            method_kwargs={
                "n_splits": 8,
                "n_trials": 5,
                "timeout": 600,
                "random_state": 0,
                "n_jobs": -1,
                "max_greedy_ensemble_size": 10,
                "greedy_tol": 5e-4,  # strict selectivity: require larger gain to add members
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
