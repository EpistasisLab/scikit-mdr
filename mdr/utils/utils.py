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
import itertools
from collections import Counter
import scipy
import numpy as np
import copy
import matplotlib.pyplot as plt
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

def two_way_information_gain(X, Y, Z, base=2):
    """Calculates the two-way information gain between three variables, I(X;Y;Z), in the given base

    IG(X;Y;Z) indicates the information gained about variable Z by the joint variable X_Y, after removing
    the information that X and Y have about Z individually. Thus, two-way information gain measures the
    synergistic predictive value of variables X and Y about variable Z.

    Parameters
    ----------
    X: array-like (# samples)
        An array of values for which to compute the 2-way information gain
    Y: array-like (# samples)
        An array of values for which to compute the 2-way information gain
    Z: array-like (# samples)
        An array of outcome values for which to compute the 2-way information gain
    base: integer (default: 2)
        The base in which to calculate 2-way information

    Returns
    ----------
    mutual_information: float
        The information gain calculated according to the equation IG(X;Y;Z) = I(X,Y;Z) - I(X;Z) - I(Y;Z)

    """
    X_Y = ['{}{}'.format(x, y) for x, y in zip(X, Y)]
    return (mutual_information(X_Y, Z, base=base) -
            mutual_information(X, Z, base=base) -
            mutual_information(Y, Z, base=base))

def three_way_information_gain(W, X, Y, Z, base=2):
    """Calculates the three-way information gain between three variables, I(W;X;Y;Z), in the given base

    IG(W;X;Y;Z) indicates the information gained about variable Z by the joint variable W_X_Y, after removing
    the information that W, X, and Y have about Z individually and jointly in pairs. Thus, 3-way information gain
    measures the synergistic predictive value of variables W, X, and Y about variable Z.

    Parameters
    ----------
    W: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    X: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    Y: array-like (# samples)
        An array of values for which to compute the 3-way information gain
    Z: array-like (# samples)
        An array of outcome values for which to compute the 3-way information gain
    base: integer (default: 2)
        The base in which to calculate 3-way information

    Returns
    ----------
    mutual_information: float
        The information gain calculated according to the equation:
            IG(W;X;Y;Z) = I(W,X,Y;Z) - IG(W;X;Z) - IG(W;Y;Z) - IG(X;Y;Z) - I(W;Z) - I(X;Z) - I(Y;Z)

    """
    W_X_Y = ['{}{}{}'.format(w, x, y) for w, x, y in zip(W, X, Y)]
    return (mutual_information(W_X_Y, Z, base=base) -
            two_way_information_gain(W, X, Z, base=base) -
            two_way_information_gain(W, Y, Z, base=base) -
            two_way_information_gain(X, Y, Z, base=base) -
            mutual_information(W, Z, base=base) -
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
        An instance of the MDR type to use.
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
        for features in itertools.combinations(range(X.shape[1]), cur_n):
            mdr_model = copy.deepcopy(mdr_instance)
            mdr_model.fit(X[:, features], y)
            mdr_model_score = mdr_model.score(X[:, features], y)
            model_features = [feature_names[feature] for feature in features]
            yield mdr_model, mdr_model_score, model_features

def plot_mdr_grid(mdr_instance):
    """Visualizes the MDR grid of a given fitted MDR instance. Only works for 2-way MDR models.
    
    This function is currently incomplete.

    Parameters
    ----------
    mdr_instance: object
        A fitted instance of the MDR type to visualize.

    Returns
    ----------
    fig: matplotlib.figure
        Figure object for the visualized MDR grid.

    """
    var1_levels = list(set([variables[0] for variables in mdr_instance.feature_map]))
    var2_levels = list(set([variables[1] for variables in mdr_instance.feature_map]))
    max_count = np.array(list(mdr_instance.class_count_matrix.values())).flatten().max()

    """
    TODO:
        - Add common axis labels
        - Make sure this scales for smaller and larger record sizes
        - Extend to 3-way+ models, e.g., http://4.bp.blogspot.com/-vgKCjEkWFUc/UPwPuHo6XvI/AAAAAAAAAE0/fORHqDcoikE/s1600/model.jpg
    """

    fig, splots = plt.subplots(ncols=len(var1_levels), nrows=len(var2_levels), sharey=True, sharex=True)
    fig.set_figwidth(6)
    fig.set_figheight(6)

    for (var1, var2) in itertools.product(var1_levels, var2_levels):
        class_counts = mdr_instance.class_count_matrix[(var1, var2)]
        splot = splots[var2_levels.index(var2)][var1_levels.index(var1)]
        splot.set_yticks([])
        splot.set_xticks([])
        splot.set_ylim(0, max_count * 1.5)
        splot.set_xlim(-0.5, 1.5)

        if var2_levels.index(var2) == 0:
            splot.set_title('X1 = {}'.format(var1), fontsize=12)
        if var1_levels.index(var1) == 0:
            splot.set_ylabel('X2 = {}'.format(var2), fontsize=12)

        bars = splot.bar(left=range(class_counts.shape[0]),
                         height=class_counts, width=0.5,
                         color='black', align='center')

        bgcolor = 'lightgrey' if mdr_instance.feature_map[(var1, var2)] == 0 else 'darkgrey'
        splot.set_axis_bgcolor(bgcolor)
        for index, bar in enumerate(bars):
            splot.text(index, class_counts[index] + (max_count * 0.1), class_counts[index], ha='center')

    fig.tight_layout()
    return fig
