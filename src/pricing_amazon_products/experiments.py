import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score

from .io import save_df
from .config import (
    MODELS_METADATA_PATH,
    MODELS_DIR,
    EVAL_METRIC, 
    SCORING, 
    TARGET, 
    N_FOLDS, 
    RANDOM_STATE
)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_label,
    model_name,
    model_family
):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING
    )

    cv_rmse_scores = -cv_scores

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = np.sqrt(EVAL_METRIC(y_train, y_train_pred))
    test_score = np.sqrt(EVAL_METRIC(y_test, y_test_pred))

    params = json.dumps(model.get_params(), default=str)

    results = pd.Series({
        "model_name": model_name,
        "model_family": model_family,
        "feature_label": feature_label,
        "target": TARGET,
        "scoring_name": SCORING,
        "params": params,
        "cv_score_mean": cv_rmse_scores.mean(),
        "cv_score_std": cv_rmse_scores.std(),
        "train_score": train_score,
        "test_score": test_score,
    })

    safe_model_name = (
        model_name.replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    model_path = MODELS_DIR / f"{safe_model_name}.pkl"
    joblib.dump(model, model_path)

    return model, results


def update_model_metadata(
    results_metadata_df: pd.DataFrame,
    new_row: pd.Series,
    force_overwrite: bool = False,
    overwrite_if_better_cv: bool = False,
) -> pd.DataFrame:
    existing_mask = (
        (results_metadata_df["model_name"] == new_row["model_name"]) &
        (results_metadata_df["feature_label"] == new_row["feature_label"]) &
        (results_metadata_df["target"] == new_row["target"]) &
        (results_metadata_df["scoring_name"] == new_row["scoring_name"])
    )

    exists = existing_mask.any()
    should_update_and_save = False

    if not exists:
        should_update_and_save = True
    else:
        old_cv_rmse = results_metadata_df.loc[existing_mask, "cv_score_mean"].iloc[0]
        new_cv_rmse = new_row["cv_score_mean"]

        should_overwrite = overwrite_if_better_cv and (new_cv_rmse < old_cv_rmse)

        if should_overwrite or force_overwrite:
            results_metadata_df = results_metadata_df.loc[~existing_mask].copy()
            should_update_and_save = True

    if should_update_and_save:
        results_metadata_df = pd.concat(
            [results_metadata_df, new_row.to_frame().T],
            ignore_index=True
        )
        results_metadata_df["params"] = results_metadata_df["params"].astype(str)
        save_df(results_metadata_df, MODELS_METADATA_PATH, overwrite=True)

    return results_metadata_df


def init_metadata_df(overwrite=False):
    metadata_cols = [
        "model_name",
        "model_family",
        "feature_label",
        "target",
        "scoring_name",
        "params",
        "cv_score_mean",
        "cv_score_std",
        "train_score",
        "test_score",
    ]
        
    MODELS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODELS_METADATA_PATH.exists() and not overwrite:
        results_metadata_df = pd.read_parquet(MODELS_METADATA_PATH)

        if list(results_metadata_df.columns) != metadata_cols:
            raise ValueError(
                "Saved metadata columns do not match the expected schema."
            )

        return results_metadata_df

    results_metadata_df = pd.DataFrame(columns=metadata_cols)
    save_df(results_metadata_df, MODELS_METADATA_PATH, overwrite=True)
    return results_metadata_df

