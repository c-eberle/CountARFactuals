import os
import sys
# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Move up two levels to the root of Adversarial-Random-Forest
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Add the root of the project to sys.path so that CountARFactuals can be found
sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np
import polars as pl
import time
import scipy
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from  CountARFactuals import utils
from experiments.experiment_utils import load_dataset
import wandb
random_seed = 0
np.random.seed(random_seed)

# Load dataset
dataset_name = "external_pdac"
X, y = load_dataset(dataset_name)
continuous_features = X.columns

# Hyperparameters to be tested in ablation
n_estimators = [1, 10, 100, 1000]
min_samples_leaf = [1, 10]
max_depth = [3, 10, 50, None]
param_combinations = list(product(n_estimators, min_samples_leaf, max_depth))

# Hyperparameters for discriminator
rf_hyperparams = {"n_estimators": 100, "min_samples_leaf": 10, "max_depth": 10}
n_synth_samples = min(int(X.height/2), 1000)


# Start new wandb run for logging
config = {"Parameter combinations": param_combinations,
          "n_synth_samples": n_synth_samples,
          "num_features": len(continuous_features),
          "dataset_name": dataset_name}
wandb.init(
    project="Master-Thesis-Augmentation-Ablation",
    config=config,
    group="experiment_1",
    job_type="eval"
)

metrics_dict = {
    "arfshuffle/time_taken": None,
    "arfshuffle/matthews_corrcoef": None,
    "arfshuffle/f1_score": None,
    "forde/time_taken": None,
    "forde/matthews_corrcoef": None,
    "forde/f1_score": None,
    "n_estimators": None,
    "min_samples_leaf": None,
    "max_depth": None
}

for random_seed in range(10):
    np.random.seed(random_seed)
    for n_est, min_leaf, max_d in param_combinations:
        arf_hyperparams = {"n_estimators": n_est, "min_samples_leaf": min_leaf, "max_depth": max_d}
        print(arf_hyperparams)
        metrics_dict["n_estimators"] = n_est
        metrics_dict["min_samples_leaf"] = min_leaf
        metrics_dict["max_depth"] = max_d
        shuffled_indices = np.random.permutation(len(X))
        X, y = X[shuffled_indices], y[shuffled_indices]
        X_real = X[:n_synth_samples]

        #naive shuffle
        X_synth = X_real.select(pl.all().shuffle())
        X_shuffle = pl.concat([X_real, X_synth])
        y_shuffle = np.array([0]*n_synth_samples + [1]*n_synth_samples)
        X_train_shuffle, X_test_shuffle, y_train_shuffle, y_test_shuffle = train_test_split(X_shuffle, y_shuffle, test_size=0.2, random_state=random_seed)

        #arf shuffle
        clf0 = RandomForestClassifier(**arf_hyperparams, random_state=random_seed) 
        clf0.fit(X_train_shuffle, y_train_shuffle)
        start = time.time()
        nodeIDs = clf0.apply(X_real)
        x_real_obs = X_real.clone()
        x_real_obs = X_real.with_columns((pl.arange(0, X_real.shape[0]).alias("obs")))
        nodeIDs_pd = pl.DataFrame(nodeIDs)
        tmp = nodeIDs_pd.clone()
        tmp = tmp.with_columns((pl.arange(0, tmp.shape[0]).alias("obs")))
        tmp = tmp.rename({col: str(i) for i, col in enumerate(tmp.columns) if col != "obs"})
        tmp = tmp.unpivot(index=["obs"], value_name="leaf", variable_name="tree") # shape [x_real.shape[0]*n_trees, 3] # shape [x_real.shape[0]*n_trees, 3]
        tmp = tmp.with_columns(pl.col("tree").cast(pl.Int64)) # cast tree column from str to int
        # match real data to trees and leaves (node id for tree)
        x_real_obs = x_real_obs.join(tmp, on="obs", how="left").drop("obs")
        # sample leaves
        tmp = tmp.drop("obs")
        tmp_sampled = tmp.sample(n=X_real.shape[0], with_replacement=True)
        tmp_sampled = tmp_sampled.group_by(pl.all()).len().rename({"len": "cnt"})
        draw_from = tmp_sampled.join(x_real_obs, on=["tree", "leaf"], how="inner")
        # sample synthetic data from leaf
        grpd =  draw_from.group_by(["tree", "leaf"], maintain_order=True)
        X_synth = [utils.sample_group(group) for _, group in grpd]
        X_synth = pl.concat(X_synth, how="vertical")
        metrics_dict["arfshuffle/time_taken"] = time.time() - start
        # Merge real and synthetic data
        X_arfshuffle = pl.concat([X_real, X_synth])
        y_arfshuffle = np.array([0]*n_synth_samples + [1]*n_synth_samples)
        arfshuffle_disc = RandomForestClassifier(**rf_hyperparams, random_state=random_seed) 
        X_train_arfshuffle, X_test_arfshuffle, y_train_arfshuffle, y_test_arfshuffle = train_test_split(X_arfshuffle, y_arfshuffle, test_size=0.2, random_state=random_seed)
        arfshuffle_disc.fit(X_train_arfshuffle, y_train_arfshuffle)

        # forge
        start = time.time()
        leaves_grouped = tmp.group_by(pl.all()).len().rename({"len": "cnt"})
        leaves_grouped = leaves_grouped.with_columns(pl.Series("wt", leaves_grouped["cnt"]/leaves_grouped["cnt"].sum()))
        draws = np.random.choice(a=range(leaves_grouped.shape[0]), p = leaves_grouped["wt"], size=n_synth_samples)
        sampled_leaves = leaves_grouped[["tree","leaf"]][draws.tolist()]
        sampled_leaves_grouped = sampled_leaves.group_by(pl.all()).len().rename({"len": "cnt"})
        # fit normal distributions in leaves
        bnds = pl.DataFrame(schema={"tree": pl.Int32, "leaf": pl.Int32, "variable": pl.Utf8})
        res_list = []
        dt = x_real_obs.clone()
        for tree in range(arf_hyperparams["n_estimators"]):
            filtered_leaves = sampled_leaves_grouped.filter(pl.col("tree") == tree).select("leaf")
            dt_filtered = dt.filter((pl.col("tree") == tree) & (pl.col("leaf").is_in(filtered_leaves["leaf"])))
            #dt = sampled_leaves.filter(sampled_leaves["tree"]==tree)
            long = dt_filtered.unpivot(index=["tree", "leaf"],  # Columns to keep as identifiers
                                        on=continuous_features)  # Columns to melt
            res = long.group_by(["tree", "leaf", "variable"]).agg([pl.col("value").mean().alias("mean"), 
                                                                    pl.col("value").std().alias("sd")])
            res = res.fill_null(0)
            res = res.with_columns((pl.col("sd") + np.finfo(np.float64).eps).alias("sd")) #add epsilon to avoid zero sd
            if res["sd"].filter(res["sd"].is_null()).shape[0] != 0:
                print(res)
            res_list.append(res)
        params = pl.concat(res_list)
        sample_params = sampled_leaves.join(params, on=["tree", "leaf"], how="inner")
        # Generate new data from mixture distribution over trees
        X_synth = pl.DataFrame({col: np.full(n_synth_samples, np.nan) for col in X_real.columns})
        for col in X_synth.columns:
            myloc = sample_params.filter(pl.col("variable") == col).select("mean").to_series()
            myscale = sample_params.filter(pl.col("variable") == col).select("sd").to_series()
            drawn_samples = scipy.stats.norm(loc=myloc, scale=myscale).rvs(size=n_synth_samples)
            X_synth = X_synth.with_columns(pl.Series(col, drawn_samples))
            del(myloc, myscale)
        metrics_dict["forde/time_taken"] = time.time() - start
        # If any features are int, cast from float to int
        X_synth = X_synth.with_columns([pl.col(column).cast(pl.Int64) for column, dtype in X_real.schema.items() if dtype == pl.Int64])
        # Merge real and synthetic data
        X_forde = pl.concat([X_real, X_synth])
        y_forde = np.array([0]*n_synth_samples + [1]*n_synth_samples)
        forde_disc = RandomForestClassifier(**rf_hyperparams, random_state=random_seed) 
        X_train_forde, X_test_forde, y_train_forde, y_test_forde = train_test_split(X_forde, y_forde, test_size=0.2, random_state=random_seed)
        forde_disc.fit(X_train_forde, y_train_forde)

        # Calculate and log performance metrics
        for method in ["arfshuffle", "forde"]:
            y_test = globals()[f"y_test_{method}"]
            X_test = globals()[f"X_test_{method}"]
            disc = globals()[f"{method}_disc"]
            for metric in [accuracy_score, f1_score, roc_auc_score, matthews_corrcoef]:
                score = metric(y_test, disc.predict(X_test))
                metrics_dict[f"{method}/{metric.__name__}"] = score
                print(f"{method} {metric.__name__}: {score}")
            wandb.log(metrics_dict) #log metrics for this method

wandb.finish()
