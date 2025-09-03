# === Import Packages ===
import pandas as pd
import numpy as np
from math import sqrt

# === Import Custom Modules ===
from notebooks.utils.SplitData import split_data
from notebooks.utils.GrandientDescent import build_model_linreg, predict_linreg
from notebooks.utils.GradientDescentUtils import get_features_targets, normalize_z
from notebooks.utils.EvaluationModelUtils import (
    r2_score, mean_squared_error, adjusted_r2_score, mean_absolute_error
)

# === Load Data ===
url = './datasets/food_wastage_data.csv'
df = pd.read_csv(url)
df.drop_duplicates(inplace=True)

# === Define Features and Target ===
independent_variable = ['Type of Food', 'Event Type', 'Preparation Method', 'Pricing', 'Quantity of Food', 'Geographical Location']
dependent_variable = ['Wastage Food Amount']

# === Extract Features and Target into Dict ===
df_feature, df_target = get_features_targets(df, independent_variable, dependent_variable)

data_dict = {
    "X": df_feature,
    "y": df_target
}

# === Train-Test Split ===
X_train, X_test, y_train, y_test = split_data(data_dict["X"], data_dict["y"], test_size=0.3, random_state=42)

# === Train Model ===
model, J_storage = build_model_linreg(X_train, y_train)

# === Predict ===
predictions = predict_linreg(
    X_test.to_numpy(), 
    model["beta"], 
    means=model["means"], 
    stds=model["stds"]
)

# Get all columns from training after one-hot encoding
encoded_train_df = pd.get_dummies(df[independent_variable], drop_first=True)
feature_columns = encoded_train_df.columns  # Save this for future use

def map_raw_input_to_one_hot(raw_input: dict, feature_columns: list[str]) -> dict:
    one_hot_input = dict.fromkeys(feature_columns, 0)  # initialize all to 0

    # Set numeric field directly
    if 'Number of Guests' in feature_columns:
        one_hot_input['Quantity of Food'] = raw_input.get('Quantity of Food', 0)

    # One-hot encode categorical fields
    for key, value in raw_input.items():
        col_name = f"{key}_{value}"
        if col_name in one_hot_input:
            one_hot_input[col_name] = 1

    return one_hot_input

def predict_food_waste(input_row, feature_columns=feature_columns, model=model):
    one_hot_input = map_raw_input_to_one_hot(input_row, feature_columns)
    input_df = pd.DataFrame([one_hot_input])
    #input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_df.reindex(columns=feature_columns, fill_value=0)

    input_np, _, _ = normalize_z(
        input_encoded.to_numpy(),
        columns_means=model["means"],
        columns_stds=model["stds"]
    )
    y_pred = predict_linreg(input_np, model["beta"], means=model["means"], stds=model["stds"])
    return y_pred[0][0]
