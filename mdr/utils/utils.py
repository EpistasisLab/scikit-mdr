# -*- coding: utf-8 -*-

"""
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
from itertools import combinations
import copy

def n_way_models(mdr_instance, X, y, n=[2], feature_names=None):
    """Fits a MDR model to all n-way combinations of the features in X.

    Note that this function performs an exhaustive search through all feature combinations and can be computationally expensive.

    Parameters
    ----------
    mdr_instance: object
        An instance of the MDR type you want to use.
    X: array-like (# rows, # features)
        NumPy matrix containing the features
    y: array-like (# rows, 1)
        NumPy matrix containing the target values
    n: list (default: [2])
        The maximum size(s) of the MDR model to generate.
        e.g., if n == [3], all 3-way models will be generated.
    feature_names: list (default: None)
        The corresponding names of the features in X.
        If None, then the features will be named according to their order.

    Returns
    ----------
    (fitted_model, fitted_model_score, fitted_model_features): tuple of (list, list, list)
        fitted_model contains the MDR model fitted to the data.
        fitted_model_score contains the training scores corresponding to the fitted MDR model.
        fitted_model_features contains a list of the names of the features that were used in the corresponding model.

    """
    if feature_names is None:
        feature_names = list(range(X.shape[1]))

    for cur_n in n:
        for features in combinations(range(X.shape[1]), cur_n):
            mdr_model = copy.deepcopy(mdr_instance)
            mdr_model.fit(X[:, features], y)
            mdr_model_score = mdr_model.score(X[:, features], y)
            model_features = [feature_names[feature] for feature in features]
            yield mdr_model, mdr_model_score, model_features
