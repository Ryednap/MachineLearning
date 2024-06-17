import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)


_TARGET_MAP = {
    "Graduate": 0,
    "Dropout": 1,
    "Enrolled": 2,
}

_TARGET_INVERSE_MAP = {
    0: "Graduate",
    1: "Dropout",
    2: "Enrolled",
}


def get_train_test_data(
    train_path: str,
    test_path: str,
    include_original=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns training and testing data after reading and performing
    operations on it.
    """

    train = pd.read_csv(train_path)
    train = train.drop("id", axis=1)

    test = pd.read_csv(test_path)

    academic_succes = fetch_ucirepo(id=697)
    original = academic_succes.data.features
    original["Target"] = academic_succes.data.targets

    train.columns = train.columns.str.lower()
    original.columns = original.columns.str.lower()
    test.columns = test.columns.str.lower()

    print(f"Train shape: {train.shape}")
    print(f"Original shape: {original.shape}")
    print(f"Test shape: {test.shape}")

    if include_original:
        train = pd.concat([train, original], ignore_index=True)

    train["target"] = train["target"].apply(lambda x: _TARGET_MAP[x])
    return train, test


def _get_fold(config) -> BaseCrossValidator:
    n_splits = config["n_splits"]
    seed = config["seed"]
    match config["split_type"]:
        case "KFold":
            return KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
        case "StratifiedKFold":
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
        case "RepeatedStratifiedKFold":
            return RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=config["n_repeats"],
                random_state=seed,
            )
        case _:
            raise ValueError(f"Unkown split type: {config['split_type']}")


best_score = 0.0


def get_HPO_cross_validation_score(config, estimator, X, y) -> float:
    fold = _get_fold(config["hpo"])

    score = 0.0
    for i, (train_index, val_index) in enumerate(fold.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

        X_train = X_train.to_numpy(dtype=np.float32)
        y_train = y_train.to_numpy(dtype=np.float32)

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_valid)

        score += skmetrics.accuracy_score(y_valid, y_pred)
        
    return score / config['hpo']['n_splits']


def generate_cross_validation(
    config, estimator, X, y, X_test
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates final fold predictions and scores 
    """
    fold = _get_fold(config["model"])
    n_splits = config['model']['n_splits']
    n_repeats = config['model'].get('n_repeats', 1)

    oof_val = np.zeros((X.shape[0], 3))
    oof_test = np.zeros((X_test.shape[0], 3))

    scores1, scores2 = [], []
    for train_index, val_index in fold.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_valid)
        y_pred_proba = estimator.predict_proba(X_valid)

        scores1.append(skmetrics.accuracy_score(y_valid, y_pred))
        scores2.append(skmetrics.roc_auc_score(y_valid, y_pred_proba, multi_class="ovo"))
       
        oof_val[val_index] += y_pred_proba

        y_test_proba = estimator.predict_proba(X_test)
        oof_test += y_test_proba
   
       
    oof_test /= (n_splits * n_repeats)
    oof_val /= n_repeats
   
    return scores1, scores2, oof_val, oof_test
