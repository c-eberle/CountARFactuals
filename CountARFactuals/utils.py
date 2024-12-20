import numpy as np 
import pandas as pd
import polars as pl
import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm, beta
from scipy.stats._continuous_distns import FitSolverError
import pickle


def bnd_fun(tree, p, forest, feature_names, feature_value_range):
    my_tree = forest.estimators_[tree].tree_
    num_nodes = my_tree.node_count
    lb = np.full(shape=(num_nodes, p), fill_value=float(feature_value_range["min"]))
    ub = np.full(shape=(num_nodes, p), fill_value=float(feature_value_range["max"]))
    for i in range(num_nodes):
        left_child = my_tree.children_left[i]
        right_child = my_tree.children_right[i]
        if left_child > -1: # leaf nodes are indicated by -1
            ub[left_child,: ] = ub[right_child, :] = ub[i,: ]
            lb[left_child,: ] = lb[right_child,: ] = lb[i,: ]
            if left_child != right_child:
                # If no pruned node, split changes bounds
                ub[left_child, my_tree.feature[i]] = lb[right_child,  my_tree.feature[i]] = my_tree.threshold[i]
    leaves = np.where(my_tree.children_left < 0)[0]
    # lower and upper bounds, to long format, to single data frame for return
    l = pl.concat([pl.DataFrame(np.full(shape=(leaves.shape[0]), fill_value=tree), schema = ["tree"]), 
                             pl.DataFrame(leaves, schema = ["leaf"]), 
                             pl.DataFrame(lb[leaves,], schema = feature_names)], how = "horizontal")
    u = pl.concat([pl.DataFrame(np.full(shape=(leaves.shape[0]), fill_value=tree), schema = ["tree"]), 
                             pl.DataFrame(leaves, schema = ["leaf"]), 
                             pl.DataFrame(ub[leaves,], schema = feature_names)], how = "horizontal")
    l_melted = l.melt(id_vars=["tree", "leaf"], value_name="min")
    u_melted = u.melt(id_vars=["tree", "leaf"], value_name="max")
    ret = l_melted.join(u_melted, on=["tree", "leaf", "variable"], how="inner")
    del(l, u, l_melted, u_melted)
    return ret
    
def prep_evi(params, evidence):
    """
    Make sure evidence is in correct format for calculating leaf posterior.

    Parameters:
    params (dict): arf.forde_params
    evidence (pl.Series): Partial sample containing values for a subset of features to condition on.

    Returns:
    return_type: pl.DataFrame with columns ["variable", "family", "value"]
    """
    # Check if evidence is a pl.Series
    if not isinstance(evidence, pl.Series):
        raise TypeError("Evidence must be a pl.Series.")
    
    if evidence.is_empty():
        return params["meta"].with_columns(pl.lit(None).alias("value"))

    # Check if evidence contains features not contained in the original data
    if set(evidence.name) - set(params["meta"]["variable"].to_list()):
        raise KeyError("Evidence contains unknown features.")

    # Turn evidence from pl.Series to pl.DataFrame
    evidence = pl.DataFrame(evidence)
    evidence = evidence.rename({evidence.columns[0]: "value"})

    # Merge metadata with evidence
    evidence = pl.merge(pl.DataFrame(params["meta"]), pl.DataFrame(evidence), on="variable", how="left")

    return evidence
    

def encode_categorical(data, encoder = OneHotEncoder()):
    """
    Encodes all categorical features in a pandas DataFrame and returns a modified copy with encoded categorical features.

    Args:
        data (pandas.DataFrame): The input DataFrame containing categorical features.
        encoder (sklearn.preprocessing.OneHotEncoder or sklearn.preprocessing.OrdinalEncoder): The encoder to use for encoding categorical features.

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with categorical features encoded as integers.
    """
    
    categorical_features = data.select_dtypes(include=['object', 'category']).columns
    if categorical_features.empty:
        return data
    encoded_data = data.clone()
    encoded_categorical_features = []

    if type(encoder) == OneHotEncoder:
        for feature in categorical_features:
            # Reset index to avoid alignment issues
            encoded_data = encoded_data.reset_index(drop=True)
            # Fit and transform the encoder on the feature
            encoded_feature = encoder.fit_transform(encoded_data[[feature]]).toarray()
            # Get the new column names for the encoded features
            encoded_feature_columns = encoder.get_feature_names_out([feature])
            encoded_categorical_features.append(encoded_feature_columns)
            # Create a DataFrame with the encoded features
            encoded_feature_df = pl.DataFrame(encoded_feature, columns=encoded_feature_columns)
            # Drop the original 'age_category' column and concatenate the encoded features
            encoded_data = encoded_data.drop(feature, axis=1).join(encoded_feature_df)
        return encoded_data, np.concatenate(encoded_categorical_features).tolist()

    elif type(encoder) == OrdinalEncoder:
        sklearn.set_config(transform_output="pandas") # Set default output to pandas

        for feature in categorical_features:
            # Reset index to avoid alignment issues
            encoded_data = encoded_data.reset_index(drop=True)
            # Fit and transform the encoder on the feature
            encoded_feature_df = encoder.fit_transform(encoded_data[[feature]])
            encoded_data = encoded_data.drop(feature, axis=1).join(encoded_feature_df)
        return encoded_data, categorical_features.values.tolist(), encoder.categories_
    else:
        raise ValueError("The encoder must be an instance of OneHotEncoder or OrdinalEncoder.")

def recode_categorical(synth, data):
    """
    Recode categorical features in synthetic data to original levels based on the input data.

    Args:
        synth (pandas.DataFrame): The synthetic data with encoded categorical features.
        data (pandas.DataFrame): The original data with unencoded categorical features.

    Returns:
        pandas.DataFrame: The synthetic data with categorical features recoded to original levels.
    """
    encoder = OneHotEncoder()
    recoded_synth = synth.clone()
    categorical_features = data.select_dtypes(include=['object', 'category']).columns
    
    for feature in categorical_features:
        # Fit the encoder on the original feature
        encoder.fit(data[[feature]])
        # Determine the encoded feature columns
        encoded_feature_columns = encoder.get_feature_names_out([feature])
        # Extract the relevant columns from the synthetic data
        encoded_feature_data = recoded_synth[encoded_feature_columns].values
        # Inverse transform to get back the original categorical values
        original_feature_data = encoder.inverse_transform(encoded_feature_data)
        # Drop the encoded feature columns from synthetic data
        recoded_synth = recoded_synth.drop(encoded_feature_columns, axis=1)
        # Add the original categorical values back to the synthetic data
        recoded_synth[feature] = original_feature_data
    
    return recoded_synth

def sample_group(group):
    n_samples = group["cnt"][0]
    group = group.drop(["cnt", "tree", "leaf"])
    # Independently sample each column
    #sampled_columns = group.select(pl.all().apply(lambda s: s.sample(n=n_samples, with_replacement=True)))
    sampled_columns = group.select([pl.col(col).sample(n=n_samples, with_replacement=True) for col in group.columns])
    return pl.DataFrame(sampled_columns)

def write_pkl(object, path):
    """
    Save object (e.g. arf_params, counterfactuals) to a .pkl file.

    Args:
        object: Object to be saved.
        path (str): The path to save the object.
    """
    with open(path, "wb") as file:
        pickle.dump(object, file)
    return

def read_pkl(path):
    """
    Load object (e.g. arf_params, counterfactuals) from a .pkl file.

    Args:
        path (str): The path to load the object from.

    Returns:
        Object: The loaded object.
    """
    with open(path, "rb") as file:
        # Load the object from the file
        object = pickle.load(file)
    return object

def fit_beta_old(group, floc=0, fscale=1):
    if group["value"].nunique() <= 1:
        print(f"Uniform data: \n{group.drop_duplicates()}\nReturning alpha=0.01, beta=10")
        return pl.Series({"alpha": 0.01, "beta": 10, "loc": floc, "scale": fscale})
    try:
        alpha_param, beta_param, loc, scale = beta.fit(group["value"], floc=floc, fscale=fscale)
        return pl.Series({
            "alpha": alpha_param,
            "beta": beta_param,
            "loc": loc,
            "scale": scale
        })
    except FitSolverError as e:
        # Handle cases where fitting fails, return NaN values to maintain consistency
        print(f"Error fitting beta distribution: {e}\nUnique feature values: {group.drop_duplicates()}")
        print(f"Most likely due to a feature with only zeros. Returning alpha=0.01, beta=10")
        return pl.Series({
            "alpha": 0.01,
            "beta": 10,
            "loc": floc,
            "scale": fscale
        })
    
def fit_beta(data, floc=0.0, fscale=100.0):
    assert isinstance(data, list), "Data must be type list."
    assert isinstance(data[0], pl.Series), "Data must contain pl.Series."
    data = data[0]
    try:
        alpha_param, beta_param, loc, scale = beta.fit(data.to_numpy(), floc=floc, fscale=fscale)
        return pl.Series([alpha_param, beta_param, loc, scale])
    except FitSolverError as e:
        # Handle cases where fitting fails, return NaN values to maintain consistency
        return pl.Series([0.01, 10.0, floc, fscale])
        

def plot_leaf_distribution(n, my_arf, leaves, true_preds=None, export_path="plots/leaf_distribution.png"):
    """
    Plot the distribution of a feature for a set of leaves.
    """
    params = pl.merge(my_arf.params, leaves, on=["tree", "nodeid"])
    params = params[["tree", "nodeid", "variable", "mean", "sd"]]
    params = params.loc[params["variable"]=="pred"]
    params.loc[:, "lower"] = 0.51
    params.loc[:, "upper"] = 1.0
    params["lik"] = norm.cdf(x=params["upper"], loc=params["mean"], scale=params["sd"]) - norm.cdf(x=params["lower"], loc=params["mean"], scale=params["sd"])
    params.loc[:, "log_lik"] = np.log(params["lik"])
    params = pl.merge(params, my_arf.bnds[['nodeid', 'cvg', 'tree']], on=['tree', 'nodeid'], how='left')
    params.loc[:, "samples_in_leaf"] = params["cvg"] * my_arf.x_real_shape[0]
    params.drop_duplicates(inplace=True)

    # Calculate the number of rows and columns
    nrows = (n + 2) // 3  # Rows based on 3 plots per row
    ncols = 3  # 3 plots per row
    # Set the figure size dynamically based on the number of rows
    fig_width = 15  # Keep width constant
    fig_height = nrows * 3  # Set height dynamically (4 units per row)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), constrained_layout=True)
    # Flatten the axes array for easier indexing
    axs = axs.flatten()
    # Loop through the first n rows of the DataFrame and plot two Gaussians per row
    for i in range(n):
        row = params.iloc[i]
        mean = row['mean']
        std_dev = row['sd']
        lik = round(row['lik'], 2)
        lower = row['lower']
        upper = row['upper']
        tree = row['tree']
        nodeid = row['nodeid']
        cvg = round(row["cvg"], 3)
        n_samples = row["samples_in_leaf"]
        true_pred = true_preds[i] if true_preds is not None else None
        # Generate x values for the Gaussian curve
        x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
        # Get the Gaussian distribution for the given mean and standard deviation
        y = norm.pdf(x, mean, std_dev)
        # Plot the Gaussian curve on the subplot
        axs[i].plot(x, y)
        # Fill the area between lower and upper limits
        x_fill = np.linspace(lower, upper, 1000)
        y_fill = norm.pdf(x_fill, mean, std_dev)
        axs[i].fill_between(x_fill, y_fill, color='lightblue', alpha=0.5, label=f'p({lower} <= pred <= {upper}) = {lik}')
        # Add a vertical red line for the true_pred value
        if true_pred is not None:
            axs[i].axvline(true_pred, color='red', linestyle='--', label='True pred')
        # Add titles and labels
        axs[i].set_title(f'Tree: {tree}, NodeID: {nodeid}, cvg: {cvg}, n_samples: {n_samples}')
        axs[i].set_xlabel('pred')
        axs[i].set_ylabel('Probability Density')
        axs[i].set_xlim(0, 1)
        axs[i].legend()
    # Remove any empty subplots if n is odd
    if n % 2 != 0:
        fig.delaxes(axs[-1])
    # Export the plot
    plt.savefig(export_path)
    #plt.show()
    return