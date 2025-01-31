import pandas  as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
from pathlib import Path
import joblib


CURRENT_DIRECTORY = Path().absolute()
DATA_DIRECTORY = CURRENT_DIRECTORY/"Data"

def load_data():
    return pd.read_csv(Path(f"{DATA_DIRECTORY}/houseprice.csv"))

def  eval_function(actual, pred):
    rmse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def constants(dataset):
    return [feat for feat in dataset.columns if dataset[feat].std() == 0]

def quasi_constants(dataset, threshold=0.998):
    quasi_feats = []

    for feat in dataset.columns:
        predominant = dataset[feat].value_counts(
            normalize=True).sort_values(ascending=False).values[0]

        if predominant > threshold:
            quasi_feats.append(feat)
    return quasi_feats
    
def correlation(dataset, threshold=0.8):
    col_corr = set()
    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = dataset.columns[i]
                col_corr.add(colname)
    return col_corr

def duplicates(dataset):
    feat_duplicates = []

    for i in range(len(dataset.columns)):
        col_1 = dataset.columns[i]   

        for col_2 in dataset.columns[i+1 :]:
            if dataset[col_1].equals(dataset[col_2]):
                feat_duplicates.append(col_2)
    return feat_duplicates

data = load_data()

# feature_categorical = [c for c in data.columns if data[c].dtypes == "O"]
# feature_numerical = [c for c in data.columns if data[c].dtypes != "O"
#                      and c != "SalePrice"]

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
numerical_features = data.select_dtypes(include=numerics).columns
data = data[numerical_features]

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=["SalePrice"], axis=1), data["SalePrice"],
    test_size=0.2, random_state=42)


feat_const = constants(X_train)
X_train = X_train.drop(columns=feat_const, axis=1)
X_test = X_test.drop(columns=feat_const, axis=1)

feat_quasi = quasi_constants(X_train)
X_train = X_train.drop(columns=feat_quasi, axis=1)
X_test = X_test.drop(columns=feat_quasi, axis=1)

feat_correlated = correlation(X_train)
X_train = X_train.drop(columns=feat_correlated, axis=1)
X_test = X_test.drop(columns=feat_correlated, axis=1)

feat_duplicates = duplicates(X_train)
X_train = X_train.drop(columns=feat_duplicates, axis=1)
X_test = X_test.drop(columns=feat_duplicates, axis=1)

rfr = RandomForestRegressor(n_estimators=10, max_depth=4)

sfs = SFS(rfr,
          n_features_to_select=10,
          direction="forward",
          tol=None,
          scoring="r2",
          cv=5, n_jobs=-1)

sfs.fit(X_train, y_train)

selected_features = sfs.get_feature_names_out()

# X_train = X_train.columns[sfs.get_support()]

model = RandomForestRegressor(n_estimators=10, max_depth=4)
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_test[selected_features])

rmse, mae, r2 = eval_function(y_test, y_pred)

os.makedirs("saved_models", exist_ok=True)

joblib.dump(value=model, filename="saved_models/regression_model.pkl")

with open("saved_models/evaluation.txt", "wt") as f:
    f.write(f"RMSE-{rmse}\nMAE-{mae}\nR2_SCORE-{r2}")

with open("saved_models/features.txt", "wt") as f:
    for feat in X_train[selected_features]:
        dtype = X_train[feat].dtypes

        # Convert NumPy types to Python types
        if dtype in ["int16", "int32", "int64"]:
            dtype = "int"
        elif dtype in ["float16", "float32", "float64"]:
            dtype = "float"
        f.write(f"{feat}: {dtype}\n")    
