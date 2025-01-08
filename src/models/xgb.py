from typing import Dict

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def _custom_eval_metrics(y_pred, dmatrix) -> Dict[str, float]:
    """
    Custom evaluation function to compute various metrics for binary classification.

    Parameters:
    -----------
    y_pred : ndarray
        Class probabilities predicted by the model.
    dmatrix : DMatrix
        XGBoost DMatrix containing the true labels.

    Returns:
    --------
    out : list of tuple
        A list of tuples where each tuple contains a metric name (str) and its value (float). Metrics include:
        - 'bacc': Balanced Accuracy
        - 'class_error': Class Error
        - 'f1': F1 score
        - 'precision_class_0': Precision for class 0
        - 'recall_class_0': Recall for class 0
        - 'precision_class_1': Precision for class 1
        - 'recall_class_1': Recall for class 1 (controls early stopping)
    """
    y_true = dmatrix.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)

    bacc = balanced_accuracy_score(y_true, y_pred_binary)
    roc_auc = float(roc_auc_score(y_true, y_pred))
    class_error = 1.0 - accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0.0)

    precision = precision_score(y_true, y_pred_binary, average=None, zero_division=0.0)
    recall = recall_score(y_true, y_pred_binary, average=None, zero_division=0.0)

    # Return a list of tuples (metric_name, value)
    out = {
        'bacc': bacc,
        'class_error': class_error,
        'precision_class_0': precision[0],
        'precision_class_1': precision[1],
        'recall_class_0': recall[0],
        'recall_class_1': recall[1],
        'f1': f1,
        'roc_auc': roc_auc,  # Last metric controls early stopping;
    }

    out = {k: float(v) for k, v in out.items()}

    return out


def custom_eval_metrics(y_pred, dmatrix):
    """
    Custom evaluation function for XGBoost model. Wrapper around `_custom_eval_metrics`.

    Parameters:
    -----------
    y_pred : ndarray
        Class probabilities predicted by the model.
    dmatrix : DMatrix
        XGBoost DMatrix containing the true labels.

    Returns:
    --------
    out : list of tuple
    """
    out = _custom_eval_metrics(y_pred, dmatrix)

    # Return a list of tuples (metric_name, value)
    out = [(k, v) for k, v in out.items()]

    return out


def fit_xgb_classifier(
    params,
    df,
    x_cols,
    y_col,
    train_idx,
    test_idx,
    eval_idx,
    num_boost_round=1000,
    early_stopping_rounds=20
):
    x_train = df.loc[train_idx, x_cols]
    y_train = df.loc[train_idx, y_col]

    x_test = df.loc[test_idx, x_cols]
    y_test = df.loc[test_idx, y_col]

    x_eval = df.loc[eval_idx, x_cols]
    y_eval = df.loc[eval_idx, y_col]

    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    deval = xgb.DMatrix(x_eval, label=y_eval, enable_categorical=True)

    evals = [(dtrain, 'train'), (dtest, 'eval')]

    train_logs = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        evals_result=train_logs,
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=custom_eval_metrics,
        maximize=True,
        verbose_eval=None
    )

    # Evaluating the model
    y_pred = model.predict(deval)
    eval_metrics = _custom_eval_metrics(y_pred, deval)

    # out = {
    #     "train_logs": train_logs,
    #     "eval_metrics": eval_metrics,
    # } | params

    out = eval_metrics | params

    return out

def fit_xgb_classifier_kf_wrapper(
    kf_idx,
    params,
    df,
    x_cols,
    y_col,
    train_idx,
    test_idx,
    eval_idx,
    num_boost_round=1000,
    early_stopping_rounds=20
):
    out = fit_xgb_classifier(
        params,
        df,
        x_cols,
        y_col,
        train_idx,
        test_idx,
        eval_idx,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    return {
        "kf_idx": kf_idx,
    } | out
