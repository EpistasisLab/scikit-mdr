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

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score

class MDRBase(BaseEstimator):

    """Base Multifactor Dimensionality Reduction (MDR) functions.

    MDR can take categorical features and binary endpoints as input, and outputs a binary constructed feature or prediction."""

    def __init__(self, tie_break=1, default_label=0):
        """Sets up the MDR algorithm for feature construction.

        Parameters
        ----------
        tie_break: int (default: 1)
            Default label in case there's a tie in a set of feature pair values 
        default_label: int (default: 0)
            Default label in case there's no data for a set of feature pair values

        Returns
        -------
        None

        """
        self.tie_break = tie_break
        self.default_label = default_label
        self.class_count_matrix = None
        self.feature_map = None

    def fit(self, features, class_labels):
        """Constructs the MDR feature map from the provided training data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        class_labels: array-like {n_samples}
            List of true class labels

        Returns
        -------
        self: A copy of the fitted model

        """
        unique_labels = sorted(np.unique(class_labels))
        if len(unique_labels) != 2:
            raise ValueError('MDR only supports binary endpoints.')

        # Count the distribution of classes that fall into each MDR grid cell
        self.class_count_matrix = defaultdict(lambda: defaultdict(int))
        for row_i in range(features.shape[0]):
            feature_instance = tuple(features[row_i])
            self.class_count_matrix[feature_instance][class_labels[row_i]] += 1
        self.class_count_matrix = dict(self.class_count_matrix)

        # Only applies to binary classification
        overall_class_fraction = float(sum(class_labels == unique_labels[0])) / class_labels.size

        # If one class is more abundant in a MDR grid cell than it is overall, then assign the cell to that class
        self.feature_map = {}
        for feature_instance in self.class_count_matrix:
            counts = self.class_count_matrix[feature_instance]
            fraction = float(counts[unique_labels[0]]) / np.sum(list(counts.values()))
            if fraction > overall_class_fraction:
                self.feature_map[feature_instance] = unique_labels[0]
            elif fraction == overall_class_fraction:
                self.feature_map[feature_instance] = self.tie_break
            else:
                self.feature_map[feature_instance] = unique_labels[1]

        return self


class MDR(MDRBase, TransformerMixin):

    """Multifactor Dimensionality Reduction (MDR) for feature construction in binary classification problems.

    MDR can take categorical features and binary endpoints as input, and outputs a binary constructed feature."""

    def transform(self, features):
        """Uses the MDR feature map to construct a new feature from the provided features.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to transform

        Returns
        ----------
        array-like: {n_samples, 1}
            Constructed features from the provided feature matrix

        """
        if self.feature_map is None:
            raise ValueError('The MDR model must be fit before transform can be called.')

        # new_feature = np.zeros(features.shape[0], dtype=np.int)
        new_feature = np.zeros(features.shape[0], dtype=np.int64)

        for row_i in range(features.shape[0]):
            feature_instance = tuple(features[row_i])
            if feature_instance in self.feature_map:
                new_feature[row_i] = self.feature_map[feature_instance]
            else:
                new_feature[row_i] = self.default_label

        return new_feature.reshape(features.shape[0], 1)

    def fit_transform(self, features, class_labels):
        """Convenience function that fits the provided data then constructs a new feature from the provided features.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        class_labels: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples, 1}
            Constructed features from the provided feature matrix

        """
        self.fit(features, class_labels)
        return self.transform(features)


class MDRClassifier(MDRBase, ClassifierMixin):

    """Multifactor Dimensionality Reduction (MDR) for binary classification problems.

    MDR can take categorical features and binary endpoints as input, and outputs a binary prediction."""

    def predict(self, features):
        """Uses the MDR feature map to construct predictions from the provided features.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to transform

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        if self.feature_map is None:
            raise ValueError('The MDR model must be fit before predict can be called.')

        # new_feature = np.zeros(features.shape[0], dtype=np.int)
        new_feature = np.zeros(features.shape[0], dtype=np.int64)

        for row_i in range(features.shape[0]):
            feature_instance = tuple(features[row_i])
            if feature_instance in self.feature_map:
                new_feature[row_i] = self.feature_map[feature_instance]
            else:
                new_feature[row_i] = self.default_label

        return new_feature

    def fit_predict(self, features, class_labels):
        """Convenience function that fits the provided data then constructs predictions from the provided features.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        class_labels: array-like {n_samples}
            List of true class labels

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.fit(features, class_labels)
        return self.predict(features)

    def score(self, features, class_labels, scoring_function=None, **scoring_function_kwargs):
        """Estimates the accuracy of the predictions from the constructed feature.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict from
        class_labels: array-like {n_samples}
            List of true class labels

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        if self.feature_map is None:
            raise ValueError('The MDR model must be fit before score can be called.')

        new_feature = self.predict(features)

        if scoring_function is None:
            return accuracy_score(class_labels, new_feature)
        else:
            return scoring_function(class_labels, new_feature, **scoring_function_kwargs)
