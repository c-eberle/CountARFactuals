import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.special import softmax
import polars as pl
import random
import gower
from . import arf, arf_improved, utils

class CountARFactualClassif:
    def __init__(self, predictor, feature_value_range=None, n_synth=20, n_iterations=50, 
                 immutable_features=None, categorical_features=None, arf=None, psi=None, 
                 rf_hyperparams=None, dist="norm", use_improved=False):
        """
        Initialize the CountARFactualClassif class.
        
        Parameters:
        - predictor: The trained model used for predictions.
        - n_synth: Number of samples drawn from the marginal distributions.
        - n_iterations: Number of iterations for generating counterfactuals.
        - immutable_features: Names of features that are not allowed to be changed.
        - categorical_features: List of column names of categorical features.
        - arf: Fitted adversarial random forest.
        - psi: Fitted forde object.
        - rf_hyperparams: dict.
        """
        self.predictor = predictor
        self.feature_value_range = feature_value_range
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.arf = arf
        self.psi = psi
        self.arf_iterations = None
        self.rf_hyperparams = rf_hyperparams
        self.dist = dist
        self.use_improved = use_improved


    def run(self, x_interest, desired_class, desired_prob, immutable_features=[], n_synth=20):
        """
        Generate counterfactuals based on the adversarial random forest method.
        
        Parameters:
        - x_interest: The instance for which counterfactuals are to be generated.
        - desired_class: The desired class label for the counterfactuals.
        - desired_prob: The desired probability range for the counterfactuals.
        
        Returns:
        - A DataFrame with the generated counterfactuals.
        """
        dat = self.predictor.data.clone()
        yhat = self.predictor.predict_proba(dat)[:, 1]
        dat = dat.with_columns(pl.Series("pred", yhat)) #define pred as the probability of class 1
        x_interest_pred = self.predictor.predict_proba(x_interest)[:, desired_class].flatten()
        x_interest = x_interest.with_columns(pl.Series("pred", x_interest_pred))
        if self.arf is None:
            # Fit an adversarial random forest model
            if self.use_improved:
                self.arf = arf_improved.arf(x=dat, feature_value_range=self.feature_value_range, categorical_features=self.categorical_features,
                                rf_hyperparams=self.rf_hyperparams)
            else:
                self.arf = arf.arf(x=dat, feature_value_range=self.feature_value_range, categorical_features=self.categorical_features,
                                rf_hyperparams=self.rf_hyperparams)
        
        if self.psi is None:
            # Fit the forde object
            self.arf.forde(dist=self.dist)
            self.psi = self.arf.forde_params
        else:
            # Update the forde object
            self.arf.forde_params = self.psi

        evidence = utils.prep_evi(self.psi, x_interest[immutable_features][0].to_series())
        synth = self.arf.forge(n=n_synth, evidence=evidence, desired_prob=desired_prob,
                                immutable_features=immutable_features).drop("pred")
        valid_counterfactuals = self.filter_valid_counterfactuals(synth, desired_class, desired_prob)

        return valid_counterfactuals
    

    def filter_valid_counterfactuals(self, synth, desired_class, desired_prob):
        """
        Filter valid counterfactuals based on the desired probability range.
        """
        valid = synth.clone()
        valid = valid.with_columns(pl.Series("proba", self.predictor.predict_proba(synth)[:, desired_class].flatten()))
        valid = valid.filter((valid["proba"]>=desired_prob[0]) & (valid["proba"]<=desired_prob[1]))
        return valid.drop("proba")

# Define the predictor class to wrap the model and its data
class Predictor:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.n_features_ = data.shape[1]
    
    def predict_proba(self, X):
        contains_categorical = any(dtype == pl.Utf8 for dtype in X.dtypes)
        if contains_categorical:
            raise ValueError("Encode categorical features before prediction.")
        return self.model.predict_proba(X)
