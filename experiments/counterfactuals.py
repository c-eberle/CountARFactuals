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
import pandas as pd
import polars as pl
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from CountARFactuals import CountARFactuals as cf
from experiments.experiment_utils import load_dataset
from experiments import metrics
import dice_ml
from CF_methods.LIME import lime_counterfactual as lime_cf
import warnings
import wandb

# Suppress a  DICE-specific PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

### Experiment hyperparams ###
desired_prob = (0.51, 1.0)  # Desired probability range
n_cfs = 10 # Number of generated counterfactuals per class
method_list = ["DICE_random", "DICE_genetic", "DICE_kdtree", "LIME-CF", "CountARFactuals"]
test_size=0.3
random_seed = 42
np.random.seed(random_seed)
dataset_name = "external_pdac"

### Load and prepare data ###
X, y = load_dataset(dataset_name)
X.columns = [s.replace(" ", "_") for s in X.columns]
continuous_features = X.columns
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

# Prepare the same dataset in pandas format for other methods
Xpd = X.to_pandas().drop(columns=[""]) if "" in X.columns else X.to_pandas()
ypd = pd.Series(y)
Xpd_train, Xpd_test, ypd_train, ypd_test = train_test_split(Xpd, ypd, test_size=test_size, random_state=random_seed)

### Prepare CountARFactuals ###
rf_hyperparams = {"n_estimators": 100, "min_samples_leaf": 10, "max_depth": 10}
rf_model = RandomForestClassifier(**rf_hyperparams, random_state=random_seed)
rf_model.fit(X_train, y_train) 
predictor = cf.Predictor(rf_model, X_train)
file_path = f"saved_params/arf_{dataset_name}.pkl"
if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        # Load the object from the file
        fitted_arf = pickle.load(file)
    psi = fitted_arf.forde_params
else:
    fitted_arf, psi = None, None
countARFactual = cf.CountARFactualClassif(
    predictor=predictor,
    arf=fitted_arf,
    psi=psi,
    rf_hyperparams = rf_hyperparams,
    dist = "norm",
    use_improved = False
)
# If no params loaded from file, run once to fit forde
if countARFactual.arf is None:
    countARFactual.run(X[0], 0, desired_prob, n_synth=1)

n_synth = 10 #Number of CF candidates to generate per sample

### Prepare DICE ###
X_full = Xpd.copy()
X_full["target"] = y
d = dice_ml.Data(dataframe=X_full, continuous_features=continuous_features, outcome_name="target")
# provide the trained ML model to DiCE's model object
m = dice_ml.Model(model=rf_model, backend="sklearn")

### Prepare LIME-CF ###
lime_cf_object = lime_cf.LimeCounterfactual(
    classifier = rf_model, 
    training_data = X_train.to_numpy(), 
    feature_names = continuous_features,
    threshold_classifier = 0.51,
    max_features = None,
    class_names = [0,1],
    time_maximum = 60,
    categorical_features = []
    )

# Start new wandb run for logging
config = {"RF hyperparams": rf_hyperparams,
          "n_cfs": n_cfs,
          "num_features": len(continuous_features),
          "dataset_name": dataset_name}
log_run=True
if log_run:
    run = wandb.init(
        project="Master-Thesis-Counterfactuals",
        config=config,
        job_type="eval"
    )

zero_samples, one_samples = Xpd_test[rf_model.predict(X_test)==0], Xpd_test[rf_model.predict(X_test)==1]
if min(n_cfs, zero_samples.shape[0], one_samples.shape[0]) < n_cfs:
    print(f"Test size too small! n_cfs set down to {n_cfs}.")
    n_cfs = min(n_cfs, zero_samples.shape[0], one_samples.shape[0])
Xpd_interest = pd.concat([Xpd_test[rf_model.predict(X_test)==0].sample(n=n_cfs, random_state=random_seed),
                          Xpd_test[rf_model.predict(X_test)==1].sample(n=n_cfs, random_state=random_seed)])
X_interest = pl.DataFrame(Xpd_interest)
cf_df = pd.DataFrame(columns=continuous_features + ["method_idx"]) #Dataframe for storing counterfactuals
valid_count = {key: 0 for key in method_list}
desired_prob = (0.51, 1.0) #only relevant for countARFactuals
for i in range(X_interest.shape[0]):
    x = Xpd_interest.iloc[[i]]
    desired_class = 0 if rf_model.predict(x) == 1 else 1 
    original_score = rf_model.predict_proba(x)[:, desired_class]
    run.log({"original_score": original_score}, step=i)
    for method_idx, method_name in zip(range(len(method_list)), method_list):
        if "DICE" in method_name:
            DICE_method = method_name.replace("DICE_", "")
            exp = dice_ml.Dice(d, m, method=DICE_method)
            start = time.time()
            # generate counterfactuals with DiCE random
            if DICE_method == "random":
                dice_exp = exp.generate_counterfactuals(x, total_CFs=1, desired_class="opposite", stopping_threshold=0.51,verbose=False, random_seed=random_seed)                
            else:
                dice_exp = exp.generate_counterfactuals(x, total_CFs=1, desired_class="opposite", stopping_threshold=0.51, verbose=False)
            run.log({f"{method_name}/time_taken": time.time()-start}, step=i)
            CF_found = (not isinstance(dice_exp._cf_examples_list[0].final_cfs_df, type(None)))
            if CF_found:
                counterfactual = dice_exp._cf_examples_list[0].final_cfs_df.drop(columns=["target"])
                metrics_dict = metrics.compute_metrics(Xpd_interest.iloc[[i]], counterfactual, rf_model, desired_class, desired_prob)
                metrics_dict = {key: value[0] for key, value in metrics_dict.items()}
                new_score = rf_model.predict_proba(counterfactual)[:, desired_class]
                run.log({f"{method_name}/new_score": new_score}, step=i)
                valid_count[method_name] += 1

            else:
                counterfactual = pd.DataFrame(np.nan, index=range(1), columns=X.columns)            
                metrics_dict = ({'L1 Distance': None, 'L2 Distance': None, 'Cosine Distance': None})
        elif method_name=="LIME-CF":
            x = Xpd_interest.iloc[[i]].to_numpy().flatten()
            start = time.time()
            counterfactual = lime_cf_object.explanation(x)
            run.log({f"{method_name}/time_taken": time.time()-start}, step=i)
            CF_found = (counterfactual["new score"] > 0.51)
            if CF_found:
                counterfactual = counterfactual["counterfactual instance"]
                metrics_dict = metrics.compute_metrics(Xpd_interest.iloc[[i]], counterfactual, rf_model, desired_class, desired_prob)
                metrics_dict = {key: value[0] for key, value in metrics_dict.items()}
                valid_count[method_name] += 1
            else:
                counterfactual = pd.DataFrame(np.nan, index=range(1), columns=X.columns) 
                metrics_dict = ({'L1 Distance': None, 'L2 Distance': None, 'Cosine Distance': None})

        elif method_name=="CountARFactuals":
            x = X_interest[i]
            start = time.time()
            cf_candidates = countARFactual.run(x, desired_class, desired_prob, n_synth=n_synth)
            run.log({f"{method_name}/time_taken": time.time()-start}, step=i)
            CF_found = cf_candidates.height!=0
            if CF_found:
                cf_distances = metrics.L2_distance(x.to_pandas(), cf_candidates.to_pandas())
                counterfactual = cf_candidates[int(cf_distances.argmin())] # choose CF with minimum L2 distance
                counterfactual = counterfactual.to_pandas().drop(columns=[""]) if "" in counterfactual.columns else counterfactual.to_pandas()
                metrics_dict = metrics.compute_metrics(Xpd_interest.iloc[[i]], counterfactual, rf_model, desired_class, desired_prob)

                metrics_dict = {key: value[0] for key, value in metrics_dict.items()}
                valid_count[method_name] += 1
            else:
                counterfactual = pd.DataFrame(np.nan, index=range(1), columns=X.columns) 
                metrics_dict = ({'L1 Distance': None, 'L2 Distance': None, 'Cosine Distance': None})
        else:
            raise ValueError(f"Method {method_name} unknown.")

        # add method name to dict keys and log results
        metrics_dict = {f"{method_name}/{key}": value for key, value in metrics_dict.items()}
        run.log(metrics_dict, step=i)
        
        # Save counterfactual to df
        counterfactual["method_idx"] = method_idx
        cf_df = pd.concat([cf_df, counterfactual], ignore_index=True)

        print(f"Iteration {i}. Valid counterfactual found: {CF_found}. {method_name}")

valid_count = {f"{key}/num_valid": value for key, value in valid_count.items()}
run.log(valid_count)
run.log({"cf_df_table": wandb.Table(dataframe=cf_df, allow_mixed_types=True),
         "orig_samples_df": wandb.Table(dataframe=Xpd_interest, allow_mixed_types=True)})
run.finish()