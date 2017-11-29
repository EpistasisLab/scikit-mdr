# -*- coding: utf-8 -*-

"""
scikit-MDR was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Tuan Nguyen (tnguyen4@swarthmore.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
from collections import defaultdict

import numpy as np 

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

from .mdr import MDR

from ._version import __version__

class MDREnsemble(BaseEstimator, ClassifierMixin):

    """Bagging ensemble of Multifactor Dimensionality Reduction (MDR) models for prediction in machine learning"""

    def __init__(self, n_estimators=100, tie_break=1, default_label=0, random_state=None):
        """Sets up the MDR ensemble

        Parameters
        ----------
        n_estimators: int (default: 100)
            Number of MDR models to include in the ensemble
        tie_break: int (default: 1)
            Default label in case there's a tie in a set of feature pair values 
        default_label: int (default: 0)
            Default label in case there's no data for a set of feature pair values
        random_state: int, RandomState instance or None (default: None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.

        Returns
        -------
        None

        """
        self.n_estimators = n_estimators
        self.tie_break = tie_break
        self.default_label = default_label
        self.random_state = random_state
        self.feature_map = defaultdict(lambda: default_label)
        self.ensemble = BaggingClassifier(base_estimator=MDR(tie_break=tie_break, default_label=default_label),
                                          n_estimators=n_estimators, random_state=random_state)

    def fit(self, features, classes):
        """Constructs the MDR ensemble from the provided training data

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None

        """
        self.ensemble.fit(features, classes)

        # Construct the feature map from the ensemble predictions
        unique_rows = list(set([tuple(row) for row in features]))
        for row in unique_rows:
            self.feature_map[row] = self.ensemble.predict([row])[0]

    def predict(self, features):
        """Uses the MDR ensemble to construct a new feature from the provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to transform

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        return self.ensemble.predict(features)

    def fit_predict(self, features, classes):
        """Convenience function that fits the provided data then constructs a new feature from the provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.ensemble.fit(features, classes)
        return self.ensemble.predict(features)

    def score(self, features, classes, scoring_function=None, **scoring_function_kwargs):
        """Estimates the accuracy of the predictions from the MDR ensemble

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict from
        classes: array-like {n_samples}
            List of true class labels

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        new_feature = self.ensemble.predict(features)

        if scoring_function is None:
            return accuracy_score(classes, new_feature)
        else:
            return scoring_function(classes, new_feature, **scoring_function_kwargs)
