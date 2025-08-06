import numpy as np
import pandas as pd
from typing import Optional

def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None, 
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out: np.ndarray = np.copy(array)

    if columns_means is None:
        columns_means = array.mean(axis=0).reshape(1, -1)
    if columns_stds is None:
        columns_stds = array.std(axis=0).reshape(1, -1)    
    out = (out - columns_means[0]) / columns_stds[0]
    return out, columns_means, columns_stds

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature = pd.get_dummies(df[feature_names], drop_first=True).astype(int)
    df_target = df[target_names].copy() if target_names else pd.DataFrame()
    return df_feature, df_target

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    ones = np.ones((np_feature.shape[0], 1))
    X = np.concatenate([ones, np_feature], axis=1)
    return X

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    result = np.matmul(X, beta)
    return result

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = df_feature.index
    test_len = int(test_size * len(indices))
    if random_state is not None:
        np.random.seed(random_state)
    test_indices = np.random.choice(indices, test_len, replace=False)
    df_target_test = df_target.loc[test_indices]
    df_feature_test = df_feature.loc[test_indices]
    df_target_train = df_target.drop(test_indices)
    df_feature_train = df_feature.drop(test_indices)
    return df_feature_train, df_feature_test, df_target_train, df_target_test


def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    y_pred = calc_linreg(X, beta)
    J = (1/(2*m)) * np.sum((y_pred - y) ** 2)
    J = np.array([[J]])
    return np.squeeze(J)
