import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def loco(df, target_col="depr_prev", top_k=None, threshold=0.0,test_size: float = 0.2):
    """
    LOCO (Leave One Covariate Out) feature selection.

    Drops each feature, retrains a Random Forest, and measures the increase in MSE.
    top_k selects the top K by importance; otherwise features with importance > threshold are returned.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    baseline = RandomForestRegressor(n_estimators=100, max_features="log2").fit(X_train, y_train)
    full_score = mean_squared_error(y_test, baseline.predict(X_test))

    importances = np.zeros(X.shape[1])
    for i in tqdm(range(X.shape[1]), desc="LOCO"):
        cols = [j for j in range(X.shape[1]) if j != i]
        m = RandomForestRegressor(n_estimators=100, max_features="log2")
        m.fit(X_train[:, cols], y_train)
        score = mean_squared_error(y_test, m.predict(X_test[:, cols]))
        importances[i] = score - full_score  # higher increase in MSE = more important

    if top_k is not None:
        selected_features = feature_names[importances.argsort()[::-1][:top_k]]
    else:
        selected_features = feature_names[importances > threshold]

    return selected_features
