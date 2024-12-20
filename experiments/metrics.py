"""
This module contains the metrics used to evaluate the performance of the models.
"""
import csv
import numpy as np
import pandas as pd

def L1_distance(x, x_cf):
    return np.sum(np.abs(x.values - x_cf.values), axis=1)

def L2_distance(x, x_cf):
    return np.sqrt(np.sum(np.square(x.values - x_cf.values), axis=1))

def cosine_distance(x, x_cf):
    return 1 - (np.dot(x_cf.values, x.values.T).flatten() / (np.linalg.norm(x.values) * np.linalg.norm(x_cf.values, axis=1)))


def mahalanobis_distance(x_cf, x_mean, cov):
    # Loop to calculate distance for each counterfactual if we have multiple
    if isinstance(x_cf, pd.DataFrame):
        distances = []
        for cf in x_cf.values:
            distances.append(np.sqrt((cf - x_mean.values).T @ np.linalg.inv(cov) @ (cf - x_mean.values)))
        return np.array(distances)
    else:
        return np.sqrt((x_mean.iloc[0].values - x_cf.iloc[0].values).T @ np.linalg.inv(cov) @ (x_mean.iloc[0] - x_cf.iloc[0]))


def coverage(x_cf, predictor, desired_class, desired_prob):
    # Get the predicted probabilities for the desired class
    predicted_probs = predictor.predict_proba(x_cf)[:, desired_class]
    
    # Check if the probabilities fall within the desired probability range
    coverage_mask = (predicted_probs >= min(desired_prob)) & (predicted_probs <= max(desired_prob))
    
    # Return the coverage mask as a Series or array
    return pd.Series(coverage_mask, index=x_cf.index).sum() / len(coverage_mask)

def compute_metrics(x, x_cf, predictor, desired_class, desired_prob):
    # Calculate performance metrics
    L1_dist = L1_distance(x, x_cf)
    L2_dist = L2_distance(x, x_cf)
    cosine_dist = cosine_distance(x, x_cf)
    #mahalanobis_dist = mahalanobis_distance(x, counterfactuals, np.identity(len(x.columns)))
    coverage_value = coverage(x_cf, predictor, desired_class, desired_prob)

    # Create a dictionary with rounded metric values
    metrics_dict = {
        "L1 Distance": np.round(L1_dist, 3),
        "L2 Distance": np.round(L2_dist, 3),
        "Cosine Distance": np.round(cosine_dist, 3)
        #"Mahalanobis Distance": np.round(mahalanobis_dist, 3),
    }
    return metrics_dict

def write_csv(metrics_dict, save_path):
    # Write metrics dictionary to a CSV file
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        
        # Iterate over the dictionary and write each metric and its value to the file
        for metric, value in metrics_dict.items():
            writer.writerow([metric, value])

def read_csv(save_path):
    # Read the metrics from the CSV file
    with open(save_path, mode='r') as file:
        reader = csv.reader(file)
        metrics_dict = {rows[0]: rows[1] for rows in reader}
        metrics_dict_formatted = {}

        for key, value in metrics_dict.items():
            # Check if the value needs to be converted (i.e., if it looks like a list in string form)
            if isinstance(value, str) and '[' in value and ']' in value:
                # Remove the brackets and split the string into individual elements
                value = value.replace('[', '').replace(']', '').strip()
                # Convert the elements into a float array
                metrics_dict_formatted[key] = np.array(list(map(float, value.split())))
            elif key == 'Coverage':  # Special case for 'Coverage'
                metrics_dict_formatted[key] = np.float64(value)
            else:
                metrics_dict_formatted[key] = value

    return metrics_dict_formatted


# validate correctness of our implementation by comparing to scipy implementation
from scipy.spatial.distance import cityblock as L1_scipy
from scipy.spatial.distance import euclidean as L2_scipy
from scipy.spatial.distance import cosine as cosine_scipy
from scipy.spatial.distance import mahalanobis as mahalanobis_scipy

def test_metrics(x, x_cf):
    """
    Test the implemented metrics against the metrics from the scipy package.
    """
    x = np.array([1, 2, 3, 4, 5])
    x_cf = np.array([2, 3, 4, 5, 6])
    cov = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])
    
    assert L1_distance(x, x_cf) == L1_scipy(x, x_cf)
    assert L2_distance(x, x_cf) == L2_scipy(x, x_cf)
    assert cosine_distance(x, x_cf) == cosine_scipy(x, x_cf)
    assert mahalanobis_distance(x, x_cf, cov) == mahalanobis_scipy(x, x_cf, cov)

    print("All tests passed.")