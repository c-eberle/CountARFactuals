# Code taken and modified from: https://github.com/yramon/LimeCounterfactual/tree/master

"""
Function for explaining classified instances using evidence counterfactuals.
"""

"""
Import libraries 
"""
from lime.lime_tabular import LimeTabularExplainer
import time
import numpy as np
import pandas as pd
import re

# Helper function
def clean_string(str):
    """Function to remove relational operator characters from the string. 
    Example: 'feature <= 0.00' becomes 'feature'. """
    
    # Regular expression pattern to extract the feature name
    pattern = r'([a-zA-Z][\w]*)'
    match = re.search(pattern, str)
    if match:
        out_str = match.group(1)
    else:
        raise ValueError("No feature name found in the string.")
    return out_str
    """
    relational_operators = ['<', '>', '=']
    for op in relational_operators:
        str = str.split(op)[0] #cut off string at op
        str = str.strip() #remove trailing whitespaces
    return str
    """

class LimeCounterfactual(object):
    """Class for generating evidence counterfactuals for classifiers on behavioral/tabular data"""
    
    def __init__(self, classifier, training_data, feature_names, 
                 threshold_classifier, max_features=30, class_names=['1', '0'], 
                 time_maximum=120, categorical_features=None):
        
        """ Init function
        
        Args:
            classifier: [function] The classification model to be explained.
            
            training_data: [numpy.array] the data used to train the model (needed for LIME explainer).
            
            feature_names: [list of strings] list containing the feature names.
            
            threshold_classifier: [float] the threshold that is used for classifying 
            instances as positive or not. When score or probability exceeds the 
            threshold value, then the instance is predicted as positive. 
            We have no default value, because it is important the user decides 
            a good value for the threshold. 
            
            max_features: [int] maximum number of features allowed in the explanation(s). If None, 
            up to all features may be perturbed.
            
            class_names: [list of string values] names of the classes.
            
            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in seconds. Default is set to 2 minutes (120 seconds).
            
            categorical_features: [list of int] indices of the categorical features.
        """
        
        self.classifier = classifier
        self.class_names = class_names
        self.max_features = max_features if max_features is not None else len(feature_names)
        self.threshold_classifier = threshold_classifier
        self.feature_names = feature_names
        self.time_maximum = time_maximum 
    
        # Instantiate the LimeTabularExplainer
        self.explainer = LimeTabularExplainer(training_data,
                                              feature_names=self.feature_names,
                                              class_names=self.class_names,
                                              categorical_features=categorical_features,
                                              discretize_continuous=True)
    
    def explanation(self, instance):
        """ Generates evidence counterfactual explanation for the instance.
        
        Args:
            instance: [numpy array] instance to explain
                        
        Returns:
            A dictionary where:
                
                explanation_set: features set to zero in counterfactual explanation.
                
                feature_coefficient_set: corresponding importance weights 
                of the features in counterfactual explanation.
                
                number_active_elements: number of active elements of 
                the instance of interest.
                                
                minimum_size_explanation: number of features in the explanation.
                
                minimum_size_explanation_rel: relative size of the explanation
                (size divided by number of active elements of the instance).
                
                time_elapsed: number of seconds passed to generate explanation.
                
                score_predicted[desired_class]: predicted score/probability of for instance.
                
                score_new[desired_class]: predicted score/probability for instance when
                removing the features in the explanation set (~setting feature
                values to zero).
                
                difference_scores: difference in predicted score/probability
                before and after removing features in the explanation.
                
                expl_lime: original explanation using LIME (all active features
                with corresponding importance weights)

                counterfactual instance: modified instance resulting in desired prediction
        """
        
        tic = time.time()  # start timer
        
        instance_sparse = instance  # In case of tabular data, no need to transform with vectorizer
        nb_active_features = np.size(instance_sparse)
        score_predicted = self.classifier.predict_proba(
            pd.DataFrame(instance_sparse.reshape(1,-1), columns=self.feature_names)
            )[0]
        desired_class = np.argmin(score_predicted)
        score_predicted = score_predicted[desired_class]
        print('Initial Score predicted: ', score_predicted)
        exp = self.explainer.explain_instance(instance_sparse, self.classifier.predict_proba, num_features=nb_active_features)
        explanation_lime = exp.as_list()
        
        if np.size(instance_sparse) != 0:
            score_new = score_predicted
            k = 0
            number_perturbed = 0
            number_perturbed_nonzero = 0
            while (score_new <= self.threshold_classifier and k != len(explanation_lime) and 
                   time.time() - tic <= self.time_maximum and number_perturbed < self.max_features):
                number_perturbed = 0
                feature_names_full_index = []
                feature_coefficient = []
                k += 1
                perturbed_instance = instance_sparse.copy()
                for feature in explanation_lime[0:k]:
                    if feature[1] > 0:
                        feature_idx = self.feature_names.index(clean_string(feature[0]))
                        number_perturbed += 1
                        number_perturbed_nonzero += 1 if perturbed_instance[feature_idx] > 0.0 else 0
                        perturbed_instance[feature_idx] = 0.0  # Set feature to zero
                        feature_names_full_index.append(feature_idx)
                        feature_coefficient.append(feature[1])
                score_old = score_new
                perturbed_instance = pd.DataFrame(perturbed_instance.reshape(1,-1), columns=self.feature_names)
                score_new = self.classifier.predict_proba(perturbed_instance)[0][desired_class]
                if score_new != score_old:
                    print(f"Iteration {k}: Score predicted: {score_new}")
                    
            if score_new > self.threshold_classifier:
                time_elapsed = time.time() - tic
                minimum_size_explanation = number_perturbed
                minimum_size_explanation_rel = number_perturbed / nb_active_features
                difference_scores = score_predicted - score_new
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = feature_names_full_index[0:number_perturbed]
                feature_coefficient_set = feature_coefficient[0:number_perturbed]
                
            else:
                minimum_size_explanation = np.nan
                minimum_size_explanation_rel = np.nan
                time_elapsed = np.nan
                difference_scores = np.nan
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = []
                feature_coefficient_set = []
                print("No counterfactual found. Try increasing the time limit or max_features.")
                return {
                    'explanation_set': explanation_set, 
                    'feature_coefficient_set': feature_coefficient_set, 
                    'number_active_elements': number_active_elements, 
                    'size explanation': minimum_size_explanation, 
                    'relative size explanation': minimum_size_explanation_rel, 
                    'time elapsed': time_elapsed, 
                    'original score': score_predicted, 
                    'new score': score_new, 
                    'difference scores': difference_scores, 
                    'explanation LIME': expl_lime,
                    'counterfactual instance': perturbed_instance,
                    'number_perturbed_nonzero': number_perturbed_nonzero
                }
                
        else: 
            minimum_size_explanation = np.nan
            minimum_size_explanation_rel = np.nan
            time_elapsed = np.nan
            difference_scores = np.nan
            number_active_elements = nb_active_features
            expl_lime = explanation_lime
            explanation_set = []
            feature_coefficient_set = []
            
        return {
            'explanation_set': explanation_set, 
            'feature_coefficient_set': feature_coefficient_set, 
            'number_active_elements': number_active_elements, 
            'size explanation': minimum_size_explanation, 
            'relative size explanation': minimum_size_explanation_rel, 
            'time elapsed': time_elapsed, 
            'original score': score_predicted, 
            'new score': score_new, 
            'difference scores': difference_scores, 
            'explanation LIME': expl_lime,
            'counterfactual instance': perturbed_instance,
            'number_perturbed_nonzero': number_perturbed_nonzero
        }