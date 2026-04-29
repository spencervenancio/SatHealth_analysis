from typing import Optional

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import Lasso, lasso_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()
DROP_COLS = ["NDVI", "NDVI_binary", "NO2_column_number_density"]

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


def LASSO(df, target_col="depr_prev", alpha=0.01, top_k=None, max_iter=1_000_000, poly_degree=None):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if poly_degree is not None:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_arr = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        print(f"Polynomial features (degree={poly_degree}): n={X_arr.shape[0]}, p={X_arr.shape[1]}")
    else:
        X_arr = X.values
        feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    if top_k is not None:
        alpha = tune_alpha(X_scaled, y, top_k, max_iter=max_iter)

    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_scaled, y)

    selected_features = feature_names[lasso.coef_ != 0]
    return selected_features


def _load_df(data_path: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(data_path) if data_path else pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def _print_features(features) -> None:
    typer.echo(f"Selected {len(features)} features:")
    for f in features:
        typer.echo(f"  {f}")


@app.command()
def lasso(
    top_k: Optional[int] = typer.Option(None, help="Select at most this many features (tunes alpha automatically)."),
    alpha: float = typer.Option(0.01, help="Regularization strength (ignored when --top-k is set)."),
    poly_degree: Optional[int] = typer.Option(None, help="Expand to polynomial features up to this degree (e.g. 3 for cubic)."),
    target_col: str = typer.Option("depr_prev", help="Target column name."),
    data_path: Optional[str] = typer.Option(None, help="Override default dataset path."),
):
    df = _load_df(data_path)
    features = LASSO(df, target_col=target_col, alpha=alpha, top_k=top_k, poly_degree=poly_degree)
    _print_features(features)


@app.command()
def mdi(
    top_k: Optional[int] = typer.Option(None, help="Select top-k features by MDI importance."),
    threshold: float = typer.Option(0.01, help="Importance threshold (ignored when --top-k is set)."),
    target_col: str = typer.Option("depr_prev", help="Target column name."),
    data_path: Optional[str] = typer.Option(None, help="Override default dataset path."),
):
    df = _load_df(data_path)
    features = MDI(df, target_col=target_col, threshold=threshold, top_k=top_k)
    _print_features(features)


if __name__ == "__main__":
    app()
