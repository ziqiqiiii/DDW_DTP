import pandas as pd

def get_column_options(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    options = {}
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            options[col] = df[col].dropna().unique().tolist()
        else:
            options[col] = [0]
    return options