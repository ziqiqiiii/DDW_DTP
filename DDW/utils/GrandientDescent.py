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
    return beta, J_storage

def build_model_linreg(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    X_train = df_feature_train.to_numpy()
    y_train = df_target_train.to_numpy().reshape(-1, 1)
    
    X_train_z, means, stds = normalize_z(X_train)
    X_train_final = prepare_feature(X_train_z)
    beta = np.zeros((X_train_final.shape[1], 1))
    beta, J_storage = gradient_descent_linreg(X_train_final, y_train, beta, alpha, iterations)

    # Create model dictionary
    model = {
        "beta": beta,
        "means": means,
        "stds": stds,
    }
    
    return model, J_storage
