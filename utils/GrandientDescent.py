import numpy as np
import pandas as pd
from typing import Optional, Any
from utils.GradientDescentUtils import normalize_z, prepare_feature,calc_linreg, compute_cost_linreg

def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    # Standardize the feature using z normalization
    array_feature,_,_ = normalize_z(array_feature, columns_means=means, columns_stds=stds)
    # Add a column of constant 1s for the intercept
    X = prepare_feature(array_feature)
    # Calculate predicted y values
    result = calc_linreg(X, beta)
    assert result.shape == (array_feature.shape[0], 1)
    return result

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray, 
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for i in range(num_iters):
        y_pred = calc_linreg(X, beta)
        gradient = (1/m) * np.matmul(X.T, (y_pred - y))
        beta = beta - alpha * gradient
        J_storage[i, 0] = compute_cost_linreg(X, y, beta)
    assert beta.shape == (X.shape[1], 1)
    assert J_storage.shape == (num_iters, 1)
    return beta, J_storage

def build_model_linreg(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)

    model: dict[str, Any] = {}

    array_feature_train_z, means, stds = normalize_z(df_feature_train.to_numpy())
    X: np.ndarray = prepare_feature(array_feature_train_z)
    target: np.ndarray = df_target_train.to_numpy()
    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

    model["beta"] = beta
    model["means"] = means
    model["stds"] = stds

    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    
    return model, J_storage
