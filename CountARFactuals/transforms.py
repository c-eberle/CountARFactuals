import numpy as np
import polars as pl

def clr(data):
    """
    Centered log-ratio transformation for a Polars DataFrame.
    Note: This function assumes that data contains no zero values, they should be replaced with a small epsilon.
    The retuned values are slightly different compared to the sklearn.compositions.clr implementation. 
    However, the difference is very small (<1e-14) and should not affect the results.
    """
    assert isinstance(data, pl.DataFrame), "Data should be a Polars DataFrame."
    
    # Compute geometric mean across each row
    # Step 1: Convert data to logarithm
    log_data = data.select([pl.col(col).log().alias(col) for col in data.columns])

    # Step 2: Compute row-wise geometric mean
    # Sum all log values row-wise and divide by number of columns to get the mean in log space
    row_sums = log_data.select(pl.sum_horizontal(*log_data.columns) / len(log_data.columns)).to_series()

    # Step 3: Calculate the CLR transformed values
    clr_transformed = log_data.select(
        [pl.col(col) - row_sums for col in log_data.columns]
    )

    return clr_transformed

def clr_inv(data):
    """
    Inverse centered log-ratio transformation for a Polars DataFrame.
    This function assumes that data has been CLR-transformed and will return values
    in the simplex (each row summing to 1).
    """
    assert isinstance(data, pl.DataFrame), "Data should be a Polars DataFrame."
    
    # Step 1: Exponentiate the CLR transformed values
    exp_data = data.select([pl.col(col).exp().alias(col) for col in data.columns])
    
    # Step 2: Normalize each row so that the sum of values in each row equals 1
    row_sums = exp_data.select(pl.sum_horizontal(*exp_data.columns)).to_series()

    clr_inv_transformed = exp_data.select(
        [pl.col(col) / row_sums for col in exp_data.columns]
    )

    return clr_inv_transformed
    #"Inverse centered log-ratio transformation."
    #raise NotImplementedError("Inverse CLR transformation is not implemented yet.")

def alr(data):
    """
    Additive log-ratio transformation.
    Note: This function assumes that data contains no zero values, they should be replaced with a small epsilon.
    """
    max_vals = data.max(axis=1)
    additive_log_transformed = np.log(data.div(max_vals, axis=0))

    return additive_log_transformed

def alr_inv(data):
    "Inverse additive log-ratio transformation."
    raise NotImplementedError("Inverse ALR transformation is not implemented yet.")