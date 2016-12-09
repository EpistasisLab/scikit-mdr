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
from collections import Counter
import scipy
import numpy as np
import copy
from ..mdr import MDR

def entropy(X, base=2):
    """Calculates the entropy, H(X), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the entropy
    base: integer (default: 2)
        The base in which to calculate entropy

    Returns
    ----------
    entropy: float
        The entropy calculated according to the equation H(X) = -sum(p_x * log p_x) for all states of X

    """
    return scipy.stats.entropy(list(Counter(X).values()), base=base)

def joint_entropy(X, Y, base=2):
    """Calculates the joint entropy, H(X,Y), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the joint entropy
    Y: array-like (# samples)
        An array of values for which to compute the joint entropy
    base: integer (default: 2)
        The base in which to calculate joint entropy

    Returns
    ----------
    joint_entropy: float
        The joint entropy calculated according to the equation H(X,Y) = -sum(p_xy * log p_xy) for all combined states of X and Y

    """
    X_Y = ['{}{}'.format(x, y) for x, y in zip(X, Y)]
    return entropy(X_Y, base=base)

def conditional_entropy(X, Y, base=2):
    """Calculates the conditional entropy, H(X|Y), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the conditional entropy
    Y: array-like (# samples)
        An array of values for which to compute the conditional entropy
    base: integer (default: 2)
        The base in which to calculate conditional entropy

    Returns
    ----------
    conditional_entropy: float
        The conditional entropy calculated according to the equation H(X|Y) = H(X,Y) - H(Y)

    """
    return joint_entropy(X, Y, base=base) - entropy(Y, base=base)

def mutual_information(X, Y, base=2):
    """Calculates the mutual information between two variables, I(X;Y), in the given base

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the mutual information
    Y: array-like (# samples)
        An array of values for which to compute the mutual information
    base: integer (default: 2)
        The base in which to calculate mutual information

    Returns
    ----------
    mutual_information: float
        The mutual information calculated according to the equation I(X;Y) = H(Y) - H(Y|X)

    """
    return entropy(Y, base=base) - conditional_entropy(Y, X, base=base)

def information_gain(X, Y, Z, base=2):
    """Calculates the information gain between three variables, IG(X;Y;Z), in the given base

    IG(X;Y;Z) indicates the information gained about variable Z by the joint variable X_Y, after removing
    the information that X and Y have about Z individually. Thus, information gain measures the synergistic
    predictive value of variables X and Y about variable Z.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the information gain
    Y: array-like (# samples)
        An array of values for which to compute the information gain
    Z: array-like (# samples)
        An array of values for which to compute the information gain
    base: integer (default: 2)
        The base in which to calculate information gain

    Returns
    ----------
    mutual_information: float
        The information gain calculated according to the equation IG(X;Y;Z) = I(X,Y;Z) - I(X;Z) - I(Y;Z)

    """
    X_Y = ['{}{}'.format(x, y) for x, y in zip(X, Y)]
    return (mutual_information(X_Y, Z, base=base) -
            mutual_information(X, Z, base=base) -
            mutual_information(Y, Z, base=base))

def _mdr_predict(X, Y, labels):
    """Fits a MDR model to variables X and Y with the given labels, then returns the resulting predictions

    This is a convenience method that should only be used internally.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    Y: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    labels: array-like (# samples)
        The class labels corresponding to features X and Y

    Returns
    ----------
    predictions: array-like (# samples)
        The predictions from the fitted MDR model

    """
    return MDR().fit_predict(np.column_stack((X, Y)), labels)

def mdr_entropy(X, Y, labels, base=2):
    """Calculates the MDR entropy, H(XY), in the given base

    MDR entropy is calculated by combining variables X and Y into a single MDR model then calculating
    the entropy of the resulting model's predictions.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    Y: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    labels: array-like (# samples)
        The class labels corresponding to features X and Y
    base: integer (default: 2)
        The base in which to calculate MDR entropy

    Returns
    ----------
    mdr_entropy: float
        The MDR entropy calculated according to the equation H(XY) = -sum(p_xy * log p_xy) for all output states of the MDR model

    """
    return entropy(_mdr_predict(X, Y, labels), base=base)

def mdr_conditional_entropy(X, Y, labels, base=2):
    """Calculates the MDR conditional entropy, H(XY|labels), in the given base

    MDR conditional entropy is calculated by combining variables X and Y into a single MDR model then calculating
    the entropy of the resulting model's predictions conditional on the provided labels.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    Y: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    labels: array-like (# samples)
        The class labels corresponding to features X and Y
    base: integer (default: 2)
        The base in which to calculate MDR conditional entropy

    Returns
    ----------
    mdr_conditional_entropy: float
        The MDR conditional entropy calculated according to the equation H(XY|labels) = H(XY,labels) - H(labels)

    """
    return conditional_entropy(_mdr_predict(X, Y, labels), labels, base=base)

def mdr_mutual_information(X, Y, labels, base=2):
    """Calculates the MDR mutual information, I(XY;labels), in the given base

    MDR mutual information is calculated by combining variables X and Y into a single MDR model then calculating
    the mutual information between the resulting model's predictions and the labels.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    Y: array-like (# samples)
        An array of values corresponding to one feature in the MDR model
    labels: array-like (# samples)
        The class labels corresponding to features X and Y
    base: integer (default: 2)
        The base in which to calculate MDR mutual information

    Returns
    ----------
    mdr_mutual_information: float
        The MDR mutual information calculated according to the equation I(XY;labels) = H(labels) - H(labels|XY)

    """
    return mutual_information(_mdr_predict(X, Y, labels), labels, base=base)

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
