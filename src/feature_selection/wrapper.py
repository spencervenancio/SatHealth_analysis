import copy

import numpy as np
from typing import Callable
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def loco(df, model, target_col="depr_prev", top_k=None, threshold=0.0,
         metric: Callable = r2_score, higher_is_better: bool = True,
         test_size: float = 0.2, random_state: int = 42):
    """
    LOCO (Leave One Covariate Out) feature selection.

    Drops each feature, retrains the model, and measures the drop in metric.
    top_k selects the top K by importance; otherwise features with importance > threshold are returned.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    baseline = copy.deepcopy(model).fit(X_train, y_train)
    full_score = metric(y_test, baseline.predict(X_test))

    importances = np.zeros(X.shape[1])
    for i in tqdm(range(X.shape[1]), desc="LOCO"):
        cols = [j for j in range(X.shape[1]) if j != i]
        m = copy.deepcopy(model)
        m.fit(X_train[:, cols], y_train)
        score = metric(y_test, m.predict(X_test[:, cols]))
        importances[i] = full_score - score if higher_is_better else score - full_score

    if top_k is not None:
        selected_features = feature_names[importances.argsort()[::-1][:top_k]]
    else:
        selected_features = feature_names[importances > threshold]

    return selected_features
