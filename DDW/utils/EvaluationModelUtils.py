import numpy as np
import pandas as pd

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    # Flatten arrays in case they are column vectors
    y = y.flatten()
    ypred = ypred.flatten()
    ss_res = np.sum((y - ypred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def adjusted_r2_score(r2: int, n: int, k: int) -> int:
    numerator: int = n - 1
    denomenator: int = n - 1 - k
    return 1 - (numerator / denomenator) * (1 - r2)

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    # Flatten arrays in case they are column vectors
    target = target.flatten()
    pred = pred.flatten()
    mse = np.mean((target - pred) ** 2)
    return mse

def mean_absolute_error(target: np.ndarray, pred: np.ndarray) -> float:
    # Flatten arrays in case they are column vectors
    target = target.flatten()
    pred = pred.flatten()
    mae = np.mean(np.abs(target - pred))
    return mae
    