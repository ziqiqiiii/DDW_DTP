import pandas as pd
import numpy as np
from typing import TypeAlias
from typing import Optional, Any    

Number: TypeAlias = int | float
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

