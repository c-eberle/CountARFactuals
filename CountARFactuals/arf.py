import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy
from scipy.stats import truncnorm, norm, beta
from . import utils
import time
from CountARFactuals import zi_beta

class arf:
  """Implements Adversarial Random Forests (ARF) in python.

  Usage:
  1. fit ARF model with arf()
  2. estimate density with arf.forde()
  3. generate data with arf.forge().

  :param x: Input data.
  :type x: pandas.Dataframe
  :param delta: Tolerance parameter. Algorithm converges when OOB accuracy is < 0.5 + `delta`, defaults to 0
  :type delta: float, optional
  :param max_iters: Maximum iterations for the adversarial loop, defaults to 10
  :type max_iters: int, optional
  :param early_stop: Terminate loop if performance fails to improve from one round to the next?, defaults to True
  :type early_stop: bool, optional
  :param verbose: Print discriminator accuracy after each round?, defaults to True
  :type verbose: bool, optional
  :param min_node_size: minimum number of samples in terminal node, defaults to 5 
  :type min_node_size: int  
  :param feature_value_range: Dictionary containing the minimum and maximum value for features. If not provided, the range is inferred from the data. Defaults to None
  :type feature_value_range: dict
  """   
  def __init__(self, x, delta = 0,  max_iters = 10, early_stop = True, verbose = True, 
               feature_value_range = None, categorical_features = None, pred_feature = "pred", 
               random_seed = 42, rf_hyperparams = {"n_estimators": 30, "min_samples_leaf": 20}, **kwargs):
    
    # start timer for testing (TODO remove timers at later stage)
    initial_fit_start = time.time()

    # assertions
    assert isinstance(x, pl.DataFrame), f"expected polars DataFrame as input, got:{type(x)}"
    assert len(set(x.columns)) == x.shape[1], f"every column must have a unique column name"
    assert max_iters >= 0, f"negative number of iterations is not allowed: parameter max_iters must be >= 0"
    assert 0 <= delta <= 0.5, f"parameter delta must be in range 0 <= delta <= 0.5"
    assert isinstance(pred_feature, str), f"expected string for pred_feature, got:{type(pred_feature)}"
    assert isinstance(rf_hyperparams, dict), f"expected dict for rf_hyperparams, got {type(rf_hyperparams)}"
    assert rf_hyperparams["min_samples_leaf"] > 0, f"minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
    assert rf_hyperparams["n_estimators"] > 0, f"number of trees in the random forest (parameter n_estimators) must be greater than zero"


    # initialize values 
    x_real = x.clone()
    self.x_real_shape = x_real.shape
    self.p = x_real.shape[1]
    self.orig_colnames = x_real.columns
    self.feature_value_range = feature_value_range if feature_value_range is not None else {"min": -np.inf, "max": np.inf}
    self.categorical_features = categorical_features
    self.continuous_features = [col for col in x_real.columns if col not in self.categorical_features]
    self.pred_feature = pred_feature
    self.random_seed = random_seed
    self.rf_hyperparams = rf_hyperparams

    # If no synthetic data provided, sample from marginals
    x_synth = x_real.select([pl.col(col_name).shuffle(seed=np.random.randint(0, 1000)) for col_name in x_real.columns])  
    
    # Merge real and synthetic data
    x = pl.concat([x_real, x_synth])
    y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
    self.x_real = x_real

    # Fit initial RF model
    clf_0 = RandomForestClassifier(**self.rf_hyperparams, **kwargs) 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.random_seed)
    clf_0.fit(x_train, y_train)

    iters = 0

    acc_0 = accuracy_score(y_test, clf_0.predict(x_test))
    acc = [acc_0]

    if verbose is True:
      print(f"Initial accuracy is {acc_0}\nExecution time for initial arf iteration: {time.time() - initial_fit_start} seconds.")

    if (acc_0 > 0.5 + delta and iters < max_iters):
      iteration_start = time.time()
      converged = False
      while (not converged): # Start adversarial loop
        # get nodeIDs
        nodeIDs = clf_0.apply(self.x_real) # shape [x_real.shape[0], n_trees] 

        # add observation ID to x_real
        x_real_obs = x_real.clone()
        x_real_obs = x_real.with_columns((pl.arange(0, x_real.shape[0]).alias("obs")))

        # Assuming nodeIDs is a list or array-like structure
        nodeIDs_pd = pl.DataFrame(nodeIDs)
        tmp = nodeIDs_pd.clone()  # You can use nodeIDs_pd instead if you don't need to clone.
        tmp = tmp.with_columns(pl.Series("obs", range(0, x_real.shape[0])))
        tmp = tmp.melt(id_vars="obs", value_name="leaf", variable_name="tree")

        # Merge x_real_obs with tmp on "obs" (like a Pandas merge)
        x_real_obs = x_real_obs.join(tmp, on="obs", how="inner")

        # Drop the 'obs' column
        x_real_obs = x_real_obs.drop("obs")
        
        tmp = tmp.drop("obs")
        tmp = tmp.sample(n=x_real.shape[0], with_replacement=True)
        tmp_counts = tmp.group_by(["tree", "leaf"]).count().rename({"count": "cnt"})
        draw_from =  tmp_counts.join(x_real_obs, on=["tree", "leaf"], how="inner")

        # sample synthetic data from leaf
        grpd =  draw_from.group_by(["tree", "leaf"], maintain_order=True)
        x_synth = [utils.sample_group(group) for _, group in grpd]
        # Convert x_synth to a pandas DataFrame
        x_synth = pl.concat(x_synth, how="vertical")
        
        # delete unnecessary objects 
        del(nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from)

        # merge real and synthetic data
        x = pl.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
        
        # discriminator
        clf_1 = RandomForestClassifier(**self.rf_hyperparams,**kwargs) #add max_depth
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.random_seed)
        clf_1.fit(x_train, y_train)

        # update iters and check for convergence
        acc_1 = accuracy_score(y_test, clf_1.predict(x_test))
        acc.append(acc_1)
        
        iters += 1
        plateau = True if early_stop is True and acc[iters] > acc[iters - 1] else False
        if verbose is True:
          print(f"Iteration number {iters} reached accuracy of {acc_1}.\nExecution time for iteration {iters}: {time.time() - iteration_start} seconds.")
        if (acc_1 <= 0.5 + delta or iters >= max_iters or plateau):
          converged = True
        else:
          clf_0 = clf_1
    self.clf = clf_0
    self.acc = acc 
        
    # Pruning
    pred = self.clf.apply(self.x_real)
    for tree_num in range(0, self.rf_hyperparams["n_estimators"]):
      tree = self.clf.estimators_[tree_num]
      left = tree.tree_.children_left
      right = tree.tree_.children_right
      leaves = np.where(left < 0)[0]

      # get leaves that are too small
      unique, counts = np.unique(pred[:, tree_num], return_counts=True)
      to_prune = unique[counts < self.rf_hyperparams["min_samples_leaf"]]

      # also add leaves with 0 obs.
      to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

      while len(to_prune) > 0:
        for tp in to_prune:
          # Find parent
          parent = np.where(left == tp)[0]
          if len(parent) > 0:
            # Left child
            left[parent] = right[parent]
          else:
            # Right child
            parent = np.where(right == tp)[0]
            right[parent] = left[parent]
        # Prune again if child was pruned
        to_prune = np.where(np.in1d(left, to_prune))[0]

  def forde(self, dist = "norm", oob = False):
    """This part is for density estimation (FORDE)

    :param dist: Distribution to use for density estimation of continuous features. Distributions implemented so far: "truncnorm", "norm", defaults to "norm"
    :type dist: str, optional
    :param oob: Only use out-of-bag samples for parameter estimation? If `True`, `x` must be the same dataset used to train `arf`, defaults to False
    :type oob: bool, optional
    :return: Return parameters for the estimated density.
    :rtype: dict
    """    
 
    self.dist = dist
    self.oob = oob

    # Get terminal nodes for all observations
    pred = self.clf.apply(self.x_real)
    
    # If OOB, use only OOB trees
    if self.oob:
      for tree in range(self.rf_hyperparams["n_estimators"]):
        idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(self.clf.estimators_[tree].random_state, self.x.shape[0], self.x.shape[0]))
        pred[np.invert(idx_oob), tree] = -1
        
    # compute leaf bounds and coverage
    bnds = pl.concat([utils.bnd_fun(tree=j, p = self.p, forest = self.clf, feature_names = self.orig_colnames, feature_value_range=self.feature_value_range) for j in range(self.rf_hyperparams["n_estimators"])])
    #bnds["f_idx"]= bnds.groupby(["tree", "leaf"]).ngroup()
    bnds_2 = pl.DataFrame()
    for t in range(self.rf_hyperparams["n_estimators"]):
        unique, freq = np.unique(pred[:, t], return_counts=True)
        vv = pl.DataFrame({
            "leaf": unique,
            "cvg": freq / pred.shape[0]
        })
        zz = bnds.filter(pl.col("tree") == t)
        merged = vv.join(zz, on="leaf", how="inner")  # Use "inner" join to match records
        bnds_2 = pl.concat([bnds_2, merged], how="vertical")
    bnds = bnds_2
    del(bnds_2)

    # set coverage for nodes with single observations to zero
    if self.continuous_features:
      bnds = bnds.with_columns(
          pl.when(pl.col("cvg") == 1/pred.shape[0])
          .then(0)
          .otherwise(pl.col("cvg"))
          .alias("cvg")
)    
    # no parameters to learn for zero coverage leaves - drop zero coverage nodes
    bnds = bnds.filter(bnds["cvg"] > 0)

    # rename leafs to nodeids
    bnds = bnds.rename({"leaf": "nodeid"})
    # save bounds to later use coverage for drawing new samples
    self.bnds= bnds

    continuous_params_start = time.time()
    # Fit continuous distribution in all terminal nodes
    self.params = pl.DataFrame()
    if self.continuous_features:
      for tree in range(self.rf_hyperparams["n_estimators"]):
        tree_iter_start = time.time()
        dt = self.x_real[self.continuous_features].clone()
        dt = dt.with_columns([pl.Series("tree", [tree] * dt.shape[0]),
                              pl.Series("nodeid", pred[:, tree])])
        if self.dist == "truncnorm":
          filtered_dt = dt.filter(pl.col("nodeid") >= 0)
          melted_dt = filtered_dt.melt(id_vars=["tree", "nodeid"],  # Columns to keep as identifiers
                                       value_vars=self.continuous_features)  # Columns to melt
          long = melted_dt.join(bnds[["tree", "nodeid", "variable", "min", "max"]], on=["tree", "nodeid", "variable"], how="left")
          filtered_long = long.filter(pl.col("variable") != "pred")
          res = filtered_long.group_by(["tree", "nodeid", "variable"]).agg([pl.col("value").mean().alias("mean"),
                                                                            pl.col("value").std().alias("sd"),
                                                                            pl.col("min").min().alias("min"),
                                                                            pl.col("max").max().alias("max")])
          res = res.with_columns((pl.col("sd") + np.finfo(np.float64).eps).alias("sd")) #add epsilon to avoid zero sd
        elif self.dist == "norm":          
          filtered_dt = dt.filter(pl.col("nodeid") >= 0)
          melted_dt = filtered_dt.melt(id_vars=["tree", "nodeid"],  # Columns to keep as identifiers
                                       value_vars=self.continuous_features)  # Columns to melt
          long = melted_dt.join(bnds[["tree", "nodeid", "variable", "min", "max"]], on=["tree", "nodeid", "variable"], how="left")
          filtered_long = long.filter(pl.col("variable") != "pred")
          res = filtered_long.group_by(["tree", "nodeid", "variable"]).agg([pl.col("value").mean().alias("mean"),
                                                                            pl.col("value").std().alias("sd")])
          res = res.with_columns((pl.col("sd") + np.finfo(np.float64).eps).alias("sd")) #add epsilon to avoid zero sd
        elif self.dist == "beta":
          filtered_dt = dt.filter(pl.col("nodeid") >= 0)
          melted_dt = filtered_dt.melt(id_vars=["tree", "nodeid"],  # Columns to keep as identifiers
                                       value_vars=self.continuous_features)  # Columns to melt
          long = melted_dt.join(bnds[["tree", "nodeid", "variable", "min", "max"]], on=["tree", "nodeid", "variable"], how="left")
          filtered_long = long.filter(pl.col("variable") != "pred")
          res = utils.fit_beta(filtered_long.group_by(["tree", "nodeid", "variable"])) #TODO implement fit_beta for polars
          res = res.with_columns([
            res["beta_params"].list.get(0).alias("alpha"),
            res["beta_params"].list.get(1).alias("beta"),
            res["beta_params"].list.get(2).alias("mean"),
            res["beta_params"].list.get(3).alias("sd")])
        elif self.dist == "zi-beta":
          filtered_dt = dt.filter(pl.col("nodeid") >= 0)
          melted_dt = filtered_dt.melt(id_vars=["tree", "nodeid"],  # Columns to keep as identifiers
                                       value_vars=self.continuous_features)  # Columns to melt
          long = melted_dt.join(bnds[["tree", "nodeid", "variable", "min", "max"]], on=["tree", "nodeid", "variable"], how="left")
          filtered_long = long.filter(pl.col("variable") != "pred")
          res = utils.fit_zibeta(filtered_long.group_by(["tree", "nodeid", "variable"])) #TODO implement fit_zibeta for polars
          res = res = res.with_columns([
            res["beta_params"].list.get(0).alias("alpha"),
            res["beta_params"].list.get(1).alias("beta"),
            res["beta_params"].list.get(2).alias("mean"),
            res["beta_params"].list.get(3).alias("sd")])

        else:
          raise ValueError("unknown distribution, make sure to enter a vaild value for dist")
          exit()
        
        # Model feature "pred" seperately using a beta distribution
        long_pred = long.filter(long["variable"] == self.pred_feature)
        long_pred = long_pred.with_columns(pl.when(pl.col("value")==0).then(1e-5).
                                           when(pl.col("value")==1).then(1-1e-5).
                                           otherwise(pl.col("value")).alias("value"))
        grouped_pred = long_pred.group_by(['tree', 'nodeid', 'variable'])
        #for group in grouped_pred:
        res_pred = grouped_pred.agg(pl.map_groups(exprs=["value"], function=utils.fit_beta, returns_scalar=False).alias("beta_params"))
        res_pred = res_pred.with_columns([
          res_pred["beta_params"].list.get(0).alias("alpha"),
          res_pred["beta_params"].list.get(1).alias("beta"),
          res_pred["beta_params"].list.get(2).alias("mean"),
          res_pred["beta_params"].list.get(3).alias("sd")]).drop("beta_params")
        if self.dist == "truncnorm" or self.dist == "norm":
          res = res.with_columns([pl.lit(None).alias(col).cast(pl.Float64) for col in ["alpha", "beta"]])
          res_pred = res_pred.select(res.columns) # reorder columns to match res
        res = pl.concat([res, res_pred], how="vertical")
        self.params = pl.concat([self.params, res])
    print(f'Execution time for tree {tree+1}/{self.rf_hyperparams["n_estimators"]}: {time.time() - tree_iter_start:.2f} seconds.')
    print(f"Execution time for continuous parameter estimation: {time.time() - continuous_params_start:.2f} seconds.")

    categorical_params_start = time.time()
    #TODO implement categorical features in polars
    # Get class probabilities in all terminal nodes
    self.class_probs = pl.DataFrame()
    if self.categorical_features:
      for tree in range(self.rf_hyperparams["n_estimators"]):
        dt = self.x_real.loc[:, self.categorical_features.iloc[:, 0].tolist()].clone()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        dt = pl.melt(dt[dt["nodeid"] >= 0], id_vars = ["tree", "nodeid"])
        long = pl.merge(left = dt, right = bnds, on = ["tree","nodeid", "variable"])
        long["count_var"] = long.groupby(["tree", "nodeid", "variable"])["variable"].transform("count")
        long["count_var_val"] = long.groupby(["tree", "nodeid", "variable", "value"])["variable"].transform("count")
        long.drop_duplicates(inplace=True)
        long["prob"] = long["count_var_val"] / long["count_var"] 
        long = long[["f_idx","tree", "nodeid", "variable", "value","prob"]]
        self.class_probs = pl.concat([self.class_probs, long])
    print(f"Execution time for categorical parameter estimation: {time.time() - categorical_params_start:.2f} seconds.")

    self.forde_params = {"cnt": self.params, "cat": self.class_probs, 
                          "forest": self.clf, "meta" : pl.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}

    return
  
  def forge(self, n, evidence = None, desired_prob = None, immutable_features = []):
    """ This part is for data generation (FORGE)

    :param n: Number of synthetic samples to generate.
    :type n: int
    :return: Returns generated data.
    :rtype: pandas.DataFrame
    """
    try:
      getattr(self, "bnds")
      getattr(self, "forde_params")
    except AttributeError:
      raise AttributeError("need density estimates to generate data -- run .forde() first!")

    if desired_prob is None and immutable_features==[]:
      # Draw random leaves with probability proportional to coverage
      unique_bnds = self.bnds[["tree", "nodeid", "cvg"]].unique(maintain_order=True).rename({"cvg": "wt"})
      unique_bnds[["wt"]] = unique_bnds[["wt"]] / self.rf_hyperparams["n_estimators"]
    else:
      unique_bnds = self.leaf_posterior(evidence, desired_prob)#, desired_leaves_cvg=unique_bnds) #TODO incorporate cvg in leaf_posterior function


    # Draw n terminal nodes
    draws = np.random.choice(a=range(unique_bnds.shape[0]), p = unique_bnds["wt"], size=n)
    sampled_trees_nodes = unique_bnds[["tree","nodeid"]][draws.tolist()].with_row_count("obs")

    # Get distributions parameters for each new obs.
    if self.continuous_features:
      obs_params = sampled_trees_nodes.join(self.params, on=["tree", "nodeid"], how="inner").sort("obs")
    # Get probabilities for each new obs.
    if self.categorical_features:
      obs_params = sampled_trees_nodes.join(self.class_probs, on=["tree", "nodeid"], how="inner").sort("obs")

    # Sample new data from mixture distribution over trees
    data_new = data_new = pl.DataFrame({colname: [None] * n for colname in self.orig_colnames})
    mutable_colidx = [i for i, col in enumerate(self.orig_colnames) if col not in immutable_features]

    # Generate mutable features
    for j in mutable_colidx:
      colname = self.orig_colnames[j]
      if colname == self.pred_feature:
        my_a = obs_params.filter(pl.col("variable") == colname).select("alpha").to_series()
        my_b = obs_params.filter(pl.col("variable") == colname).select("beta").to_series()
        drawn_samples = beta.rvs(a=my_a, b=my_b, size = n)
        data_new = data_new.with_columns(pl.Series(colname, drawn_samples))
      elif colname in self.categorical_features:
        raise NotImplementedError("Categorical features not yet implemented")
      else:
        if self.dist == "truncnorm":         
         # sample from truncated normal distribution
         # note: if sd == 0, truncnorm will return location parameter -> this is desired; if we have 
         # all obs. in that leave having the same value, we sample a new obs. with exactly that value as well
         myclip_a = obs_params.filter(pl.col("variable") == colname).select("min").to_series()
         myclip_b = obs_params.filter(pl.col("variable") == colname).select("max").to_series()
         myloc = obs_params.filter(pl.col("variable") == colname).select("mean").to_series()
         myscale = obs_params.filter(pl.col("variable") == colname).select("sd").to_series()
        
         data_new.isetitem(j, truncnorm(a =(myclip_a - myloc) / myscale, b = (myclip_b - myloc) / myscale, loc = myloc, scale = myscale ).rvs(size = n))
         drawn_samples = truncnorm(a =(myclip_a - myloc) / myscale, b = (myclip_b - myloc) / myscale, 
                                               loc = myloc, scale = myscale ).rvs(size = n)
         del(myclip_a,myclip_b,myloc,myscale)

        elif self.dist =="norm":
          myloc = obs_params.filter(pl.col("variable") == colname).select("mean").to_series()
          myscale = obs_params.filter(pl.col("variable") == colname).select("sd").to_series()
          
          drawn_samples = norm(loc=myloc, scale=myscale).rvs(size=n)
          del(myloc, myscale)

        else:
          raise ValueError("Distribution not yet implemented")
        data_new = data_new.with_columns(pl.Series(colname, drawn_samples))

    # Set immutable features
    if immutable_features:
      # TODO implement this for polars
      immutable_colidx = [self.orig_colnames.index(col) for col in immutable_features]
      data_new[immutable_colidx] = evidence[0][evidence[0]["variable"].isin(immutable_features)]["value"].clone()
    return data_new


  # Compute leaf posterior
  def leaf_posterior(self, evidence=None, desired_prob=None, cvg_weight=0, lik_weight=1):
      """
      Returns a posterior distribution over leaves, conditional on some evidence (immutable features, conditional features, ...).
      """
      
      # Initialize variables to store likelihoods for continuous and categorical features
      psi_cnt = psi_cat = None

      # Handling likelihood computation for continuous features
      if any(evidence["family"] != "multinom"):
          # Filter evidence for continuous features
          evi = evidence.filter(evidence["family"] != "multinom")
          # Merge evidence with parameters for continuous features
          psi_cnt = evi.join(self.forde_params["cnt"], on="variable", how="left")
          
          # If immutable features are provided, get likelihoods for continuous features
          # Check if all values in the "value" column of evidence are NaN
          if not evidence["value"].is_null().all():
            # Convert the "value" column in psi_cnt to numeric (invalid values become NaN)
            psi_cnt = psi_cnt.with_columns(psi_cnt["value"].cast(pl.Float64))
            if self.dist == "truncnorm":
              # Compute likelihoods for equality relations (immutable features)
              psi_cnt["lik"] = psi_cnt.with_columns(
                pl.struct(["value", "min", "max", "mean", "sd"]).map_elements(
                  lambda row: truncnorm.pdf(
                    row["value"],
                    a=(row["min"] - row["mean"]) / (row["sd"] + np.finfo(float).eps),
                    b=(row["max"] - row["mean"]) / (row["sd"] + np.finfo(float).eps),
                    loc=row["mean"],
                    scale=row["sd"] + np.finfo(float).eps,),
                return_dtype=pl.Float64).alias("lik"))
            elif self.dist == "norm":
              psi_cnt = psi_cnt.with_columns(
                  (pl.col("value") - pl.col("mean")) / (pl.col("sd") + np.finfo(float).eps)
                  .map_elements(lambda x: norm.pdf(x, loc=0, scale=1), return_dtype=pl.Float64)
                  .alias("lik")
              )
          
          # Compute likelihood for desired pred probability range
          psi_cnt_pred = psi_cnt.filter(pl.col("variable") == "pred")         
          pred_lower = pl.Series([desired_prob[0]] * psi_cnt_pred.height)
          pred_upper = pl.Series([desired_prob[1]] * psi_cnt_pred.height)
          pred_lik = beta.cdf(
            pred_upper, a=psi_cnt_pred["alpha"], b=psi_cnt_pred["beta"], loc=psi_cnt_pred["mean"], 
            scale=psi_cnt_pred["sd"]+np.finfo(float).eps) - beta.cdf(
                pred_lower, a=psi_cnt_pred["alpha"], b=psi_cnt_pred["beta"], loc=psi_cnt_pred["mean"], 
                scale=psi_cnt_pred["sd"]+np.finfo(float).eps)
          psi_cnt_pred = psi_cnt_pred.with_columns(pl.Series("lik", pred_lik))

          if not any(evidence["family"] != "multinom"):
            psi_cnt = pl.concat([psi_cnt.filter(pl.col("value").is_not_null()), psi_cnt_pred]).unique()
          else:
            psi_cnt = psi_cnt_pred

          # Compute log likelihoods (note: log(0) = -inf, so replace 0 with smallest positive float)
          psi_cnt = psi_cnt.with_columns(pl.Series("log_lik", np.log(psi_cnt["lik"].replace(0, np.finfo(float).eps))))

          # Sum log likelihoods of all features to get joint likelihood of evidence within a leaf
          psi_cnt = psi_cnt.group_by(["tree", "nodeid"]).agg(pl.col("log_lik").sum().alias("log_lik"))

          psi = psi_cnt.clone()
          del psi_cnt

      # TODO Handling likelihood computation for categorical features
      
      # Merge with coverage from forest parameters.
      bnds_clone = self.bnds.clone().unique()
      bnds_clone = bnds_clone[["tree", "nodeid", "cvg"]].unique()
      psi = psi.join(bnds_clone, on=["tree", "nodeid"], how="inner")
      
      # Compute weights based on log likelihoods and coverage
      cvg_weight = cvg_weight + np.finfo(float).eps
      lik_weight = lik_weight + np.finfo(float).eps
      # Note: In the original R code, they use log(cvg * lik) but this is more efficient and numerically stable
      psi = psi.with_columns(pl.Series("wt", np.exp(np.log(cvg_weight) + np.log(psi["cvg"]) + np.log(lik_weight) + psi["log_lik"])))

      # Select unique combinations of feature index and weight
      out = psi.filter(pl.col("wt") > 0).select(["tree", "nodeid", "wt"]).unique()
      
      # Handle cases where all leaves have zero weight
      if len(out) == 0:
          print("All leaves have zero likelihood. This is probably because evidence contains an (almost) impossible combination. For categorical data, consider setting alpha>0 in forde().")
          out = psi[["tree", "nodeid"]].drop_duplicates()
          out["wt"] = 1
      
      # Normalize weights
      out = out.with_columns((pl.col("wt")/pl.col("wt").sum()).alias("wt"))
      return out

