# Imports
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests



def corr_screening(df:pd.DataFrame, target_col: str = "depr_prev", selection_criterion: str = "p_value",
                   corr_threshold: float = 0.2, return_df: bool = False, alpha: float = 0.05, 
                   top_K_features: int = 10):
    """Performs correlation screening for feature selection.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        selection_criterion (str, optional): The criterion for feature selection ("correlation", "p_value", or "top_K").
                                             Defaults to "p_value".
        corr_threshold (float, optional): The correlation threshold for feature selection. Defaults to 0.2.
        return_df (bool, optional): Whether to return the DataFrame with all features and their statistics. Defaults to False.
        alpha (float, optional): The significance level for multiple testing correction. Defaults to 0.05.
        top_K_features (int, optional): The number of top features to select. Defaults to 10.

    Returns:
        selected_features: The selected features
    """
    feature_names = df.drop(columns=[target_col]).columns
    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    p = X.shape[1]

    corrs = np.zeros(p)
    pvals = np.zeros(p)
    for j in range(p):
        corrs[j], pvals[j] = stats.pearsonr(X[:, j], y)

    reject, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': corrs,
        'p_value': pvals,
        'p_value_adj': pvals_adj,
        'reject_null': reject
    }).sort_values(by='correlation', key=abs, ascending=False)
    
    if selection_criterion == "correlation":
        selected_features = corr_df.query("abs(correlation) >= @corr_threshold")['feature']
    elif selection_criterion == "p_value":
        selected_features = corr_df.query("reject_null == True")['feature']
    elif selection_criterion == "top_K":
        selected_features = corr_df['feature'].head(top_K_features)

    return corr_df if return_df else selected_features