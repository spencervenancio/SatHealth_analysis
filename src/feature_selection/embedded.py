from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def MDI(df, target_col="depr_prev", threshold=0.01, top_k=None, n_estimators=100, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)

    importances = rf.feature_importances_
    if top_k is not None:
        selected_features = X.columns[importances.argsort()[::-1][:top_k]]
    else:
        selected_features = X.columns[importances >= threshold]

    return selected_features


def LASSO(df, target_col="depr_prev", alpha=0.01, top_k=None, max_iter=100_000):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_scaled, y)

    if top_k is not None:
        selected_features = X.columns[abs(lasso.coef_).argsort()[::-1][:top_k]]
    else:
        selected_features = X.columns[lasso.coef_ != 0]

    return selected_features

