import polars as pl
import numpy as np
from CountARFactuals.transforms import clr, clr_inv
from sklearn.datasets import fetch_california_housing, load_diabetes

# Define dataset parameters
external_pdac = {"file_path": "data/external_pdac.csv", 
                 "target": "Disease",
                 "target_encoding": {"0": "Healthy"},
                 "features_to_drop": [],
                 "dataset_name": "external_pdac"}

data_diet = {"file_path": "data/data_diet_filtered.csv", 
                 "target": "disease",
                 "target_encoding": {"0": "healthy"},
                 "features_to_drop": ["index", "subject_id", "disease", "country", "gender", "diet", "age_category"],
                 "dataset_name": "data_diet"}

winequality_red = {"file_path": "data/winequality-red.csv", 
                 "target": "quality",
                 "target_encoding": None,
                 "features_to_drop": [],
                 "dataset_name": "winequality_red"}

def load_dataset(dataset_name, clr_transform=True):
    top100 = False
    if "top100" in dataset_name:
        dataset_name = dataset_name.replace("_top100", "")
        top100 = True
    if dataset_name in ["california_housing", "diabetes"]:
        return load_dataset_sklearn(dataset_name)
    try:
        dataparam_dict = globals()[dataset_name]
    except ValueError:
        print(f"dataset unknown! provided dataset_name: {dataset_name}")

    
    file_path = dataparam_dict["file_path"]
    data = pl.read_csv(file_path, low_memory=False)
    if "" in data.columns:
        data = data.drop("")
    if "wine" in file_path:
        data = pl.read_csv(file_path, separator=';', ignore_errors=True)
        data = data.drop_nulls()

    X = data.drop(dataparam_dict["target"])
    if dataparam_dict["features_to_drop"]:
        X = data.drop(dataparam_dict["features_to_drop"])
    if clr_transform and not top100:
        continuous_features = [col for col in X.columns if X.schema[col] == pl.Float64]
        # Scale such that features add to 1 instead of 100
        X[continuous_features] = X[continuous_features] / 100
        # Add a small epsilon to the data to avoid zero values
        X[continuous_features] = X[continuous_features] + np.finfo(float).eps
        # Apply the clr transformation
        X[continuous_features] = clr(X[continuous_features])

    if dataparam_dict["target"] not in data.columns:
        raise ValueError(f"target feature {dataparam_dict['target']} not contained in data!")
    y = data.select(dataparam_dict["target"]).to_numpy().ravel()
    if dataparam_dict["target_encoding"]:
        y = data.select(
            pl.when(pl.col(dataparam_dict["target"]) == dataparam_dict["target_encoding"]["0"])
            .then(0)
            .otherwise(1)
            .alias(dataparam_dict["target"])
        ).to_numpy().ravel()
    if "wine" in file_path:
        y = data.select(
            pl.when(pl.col(dataparam_dict["target"]) <= pl.col(dataparam_dict["target"]).median())
            .then(0)
            .otherwise(1)
            .alias(dataparam_dict["target"])
        ).to_numpy().ravel()

    if top100:
        # Filter for top 100 most prevalent features
        top100_features = X.sum().transpose(include_header=True).sort(by="column_0", descending=True)[:100]["column"].to_list()
        # Transform continuous features
        continuous_features = [col for col in X.columns if X.schema[col] == pl.Float64]
        # Scale such that features add to 1 instead of 100 and add small epsilon
        X[continuous_features] = (X[continuous_features] / 100) + np.finfo(float).eps
        X[continuous_features] = clr(X[continuous_features])
        # Filter for top 100 features
        X = X[top100_features]
        if not clr_transform:
            X = clr_inv(X[top100_features])
    return X, y


def load_dataset_sklearn(dataset_name):
    if dataset_name=="california_housing":
        data = fetch_california_housing()
    else:
        data = load_diabetes()
    X = pl.DataFrame(data.data, schema=data.feature_names)
    y = pl.Series("target", data.target)
    continuous_features = X.columns

    # Binarize target based on median house value
    median_y = np.median(data.target)
    y = (y > median_y).cast(int)  # 1 if above median, 0 if below
    return X, y