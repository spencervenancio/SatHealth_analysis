import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def MDI(df, target_col="depr_prev", threshold=0.01, top_k=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    rf = RandomForestRegressor(n_estimators=100, max_features="log2")
    rf.fit(X, y)

    importances = rf.feature_importances_
    if top_k is not None:
        selected_features = X.columns[importances.argsort()[::-1][:top_k]]
    else:
        selected_features = X.columns[importances >= threshold]

    return selected_features


def tune_alpha(X_scaled, y, k, n_alphas=200, max_iter=1_000_000):
    """
    Find the largest alpha such that LASSO selects at most k features.
    Uses the full regularization path so only one path solve is needed.
    Raises ValueError if the path never reaches k features.
    """
    alphas, coefs, _ = lasso_path(X_scaled, y, n_alphas=n_alphas, max_iter=max_iter)
    n_nonzero = (coefs != 0).sum(axis=0)
    valid = np.where(n_nonzero <= k)[0]
    if valid.size == 0:
        raise ValueError(f"LASSO path never reaches k={k} features.")
    return alphas[valid.max()]  # largest index still ≤ k features = least-regularized valid alpha


def LASSO(df, target_col="depr_prev", alpha=0.01, top_k=None, max_iter=1_000_000):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if top_k is not None:
        alpha = tune_alpha(X_scaled, y, top_k, max_iter=max_iter)

    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_scaled, y)

    selected_features = X.columns[lasso.coef_ != 0]
    return selected_features

