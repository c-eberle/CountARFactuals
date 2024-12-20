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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from  CountARFactuals import utils
from experiments.experiment_utils import load_dataset
from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE
import wandb
random_seed = 0
np.random.seed(random_seed)

# Define SMOTE variants to be used
smote_methods = {"smote": SMOTE, 
                 "kmsmote": KMeansSMOTE, 
                 "svmsmote": SVMSMOTE}

# Load dataset
dataset_name = "external_pdac"
X, y = load_dataset(dataset_name)
continuous_features = X.columns

# Hyperparameters to be tested in ablation
param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=random_seed, max_iter=1000),
        'params': {
            'penalty': ['l2'],           # Regularization type
            'C': [0.01, 0.1, 1, 10, 100]  # Inverse of regularization strength
        }
    },
    'Gradient Boosting Classifier': {
        'model': GradientBoostingClassifier(random_state=random_seed),
        'params': {
            'n_estimators': [50, 100, 200],     # Number of boosting stages
            'learning_rate': [0.01, 0.1, 0.2], # Step size shrinkage
            'max_depth': [3, 5, 10]            # Maximum depth of the trees
        }
    },
    'MLP Classifier': {
        'model': MLPClassifier(random_state=random_seed, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Number of neurons in layers
            'activation': ['relu', 'tanh'],                  # Activation function
            'solver': ['adam', 'sgd'],                       # Solver for optimization
            'alpha': [0.0001, 0.001, 0.01]                   # Regularization term
        }
    }
}

# Start new wandb run for logging
n_synth_samples = min(int(X.height/2), 1000)
rf_hyperparams = {"n_estimators": 100, "min_samples_leaf": 10, "max_depth": 10}
config = {"Parameter combinations": param_grids,
          "n_synth_samples": n_synth_samples,
          "num_features": len(continuous_features),
          "dataset_name": dataset_name}
wandb.init(
    project="Master-Thesis-Augmentation-Ablation",
    config=config,
    group="experiment_1",
    job_type="eval"
)

for model_name, config in param_grids.items():
    model = config['model']
    param_grid = config['params']
    
    #wandb.log(rf_hyperparams)
    shuffled_indices = np.random.permutation(len(X))
    X, y = X[shuffled_indices], y[shuffled_indices]
    X_real = X[:n_synth_samples]

    # smote variants
    num_ones, num_zeros = (pl.Series(y).sum(), pl.Series(y).count()-pl.Series(y).sum())
    sampling_strategy = {0: int(num_zeros + n_synth_samples/2 + 100),
                         1: int(num_ones + n_synth_samples/2) + 100}
    for method, fun  in smote_methods.items():
        sampler = fun(sampling_strategy=sampling_strategy, random_state=random_seed)
        if method=="kmsmote" and dataset_name=="winequality_red":
            sampler = fun(sampling_strategy=sampling_strategy, random_state=random_seed, cluster_balance_threshold=0.1)
        start = time.time()
        X_resampled, _ = sampler.fit_resample(X.to_pandas(), y)
        X_synth = pl.DataFrame(X_resampled).join(X, on=X.columns, how="anti")
        wandb.log({f"{method}/time_taken": time.time() - start})
        assert X_synth.height>= n_synth_samples, f"method didn't generate enough synthetic samples ({X_synth.height})"
        X_sampled = pl.concat([X_real, X_synth[:n_synth_samples]])
        y_sampled = np.array([0]*n_synth_samples + [1]*n_synth_samples) # label 0 for real samples, 1 for counterfactuals
        globals()[f"X_train_{method}"], globals()[f"X_test_{method}"], globals()[f"y_train_{method}"], globals()[f"y_test_{method}"] = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=random_seed)
        globals()[f"{method}_disc"] = GridSearchCV(estimator=model, 
                                                   param_grid=param_grid,
                                                   cv=5,                  # 5-fold cross-validation
                                                   scoring='accuracy',    # Optimize for accuracy
                                                   n_jobs=-1,             # Use all CPU cores
                                                   verbose=2              # Print progress
                                                   )
        globals()[f"{method}_disc"].fit(globals()[f"X_train_{method}"], globals()[f"y_train_{method}"])
        wandb.log({f"{method}_disc.best_estimator_": globals()[f"{method}_disc"].best_estimator_.get_params()})

    #naive shuffle
    start = time.time()
    X_synth = X_real.select(pl.all().shuffle())
    wandb.log({"shuffle/time_taken": time.time() - start})
    X_shuffle = pl.concat([X_real, X_synth])
    y_shuffle = np.array([0]*n_synth_samples + [1]*n_synth_samples)
    shuffle_disc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    X_train_shuffle, X_test_shuffle, y_train_shuffle, y_test_shuffle = train_test_split(X_shuffle, y_shuffle, test_size=0.2, random_state=random_seed)
    shuffle_disc.fit(X_train_shuffle, y_train_shuffle)
    wandb.log({"shuffle_disc.best_estimator_": shuffle_disc.best_estimator_.get_params()})

    #arf shuffle
    clf0 = RandomForestClassifier(**rf_hyperparams, random_state=random_seed) 
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
    wandb.log({"arfshuffle/time_taken": time.time() - start})
    # Merge real and synthetic data
    X_arfshuffle = pl.concat([X_real, X_synth])
    y_arfshuffle = np.array([0]*n_synth_samples + [1]*n_synth_samples)
    arfshuffle_disc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2) 
    X_train_arfshuffle, X_test_arfshuffle, y_train_arfshuffle, y_test_arfshuffle = train_test_split(X_arfshuffle, y_arfshuffle, test_size=0.2, random_state=random_seed)
    arfshuffle_disc.fit(X_train_arfshuffle, y_train_arfshuffle)
    wandb.log({"arfshuffle_disc.best_estimator_": arfshuffle_disc.best_estimator_.get_params()})

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
    for tree in range(100): #range(rf_hyperparams["n_estimators"]):
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
    wandb.log({"forde/time_taken": time.time() - start})
    # If any features are int, cast from float to int
    X_synth = X_synth.with_columns([pl.col(column).cast(pl.Int64) for column, dtype in X_real.schema.items() if dtype == pl.Int64])
    # Merge real and synthetic data
    X_forde = pl.concat([X_real, X_synth])
    y_forde = np.array([0]*n_synth_samples + [1]*n_synth_samples)
    forde_disc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    X_train_forde, X_test_forde, y_train_forde, y_test_forde = train_test_split(X_forde, y_forde, test_size=0.2, random_state=random_seed)
    forde_disc.fit(X_train_forde, y_train_forde)
    wandb.log({"forde_disc.best_estimator_": forde_disc.best_estimator_.get_params()})
    # real data (baseline)
    X_realdata = X[:n_synth_samples*2]
    y_realdata = np.array([0]*n_synth_samples + [1]*n_synth_samples)
    realdata_disc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    X_train_realdata, X_test_realdata, y_train_realdata, y_test_realdata = train_test_split(X_realdata, y_realdata, test_size=0.2, random_state=random_seed)
    realdata_disc.fit(X_train_realdata, y_train_realdata)
    wandb.log({"realdata_disc.best_estimator_": realdata_disc.best_estimator_.get_params()})
    
    # Calculate and log performance metrics
    for method in list(smote_methods.keys()) + ["shuffle", "arfshuffle", "forde", "realdata"]:
        metrics_dict = {}
        y_test = globals()[f"y_test_{method}"]
        X_test = globals()[f"X_test_{method}"]
        disc = globals()[f"{method}_disc"]
        for metric in [accuracy_score, f1_score, roc_auc_score, matthews_corrcoef]:
            score = metric(y_test, disc.predict(X_test))
            metrics_dict[f"{method}/{metric.__name__}"] = score
            print(f"{method} {metric.__name__}: {score}")
        wandb.log(metrics_dict) #log metrics for this method

wandb.finish()
