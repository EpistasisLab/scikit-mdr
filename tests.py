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

from mdr import MDR, MDRClassifier, ContinuousMDR
import numpy as np
import random
import warnings
import inspect
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

def test_mdr_init():
    """Ensure that the MDR instantiator stores the MDR variables properly"""

    mdr_obj = MDR() 

    assert mdr_obj.tie_break == 1
    assert mdr_obj.default_label == 0
    assert mdr_obj.class_count_matrix is None
    assert mdr_obj.feature_map is None

    mdr_obj2 = MDR(tie_break=1, default_label=2)

    assert mdr_obj2.tie_break == 1 
    assert mdr_obj2.default_label == 2
    assert mdr_obj.class_count_matrix is None
    assert mdr_obj.feature_map is None

def test_mdr_fit():
    """Ensure that the MDR 'fit' function constructs the right matrix to count each class, as well as the right map from feature instances to labels"""
    features = np.array([   [2,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [1,    1],
                            [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDR() 
    mdr.fit(features, classes)

    assert len(mdr.class_count_matrix) == 4
    assert len(mdr.feature_map) == 4

    assert mdr.class_count_matrix[(2, 0)][1] == 1
    assert mdr.class_count_matrix[(0, 0)][0] == 3
    assert mdr.class_count_matrix[(0, 0)][1] == 6
    assert mdr.class_count_matrix[(1, 1)][0] == 2
    assert mdr.class_count_matrix[(0, 1)][1] == 3

    assert mdr.feature_map[(2, 0)] == 1
    assert mdr.feature_map[(0, 0)] == 1
    assert mdr.feature_map[(1, 1)] == 0
    assert mdr.feature_map[(0, 1)] == 1

# 2 0 count: 1 label 1; maps to 1 
# 0 0 count: 3 label 0; 6 label 1; maps to 1 *tie_break*
# 1 1 count: 2 label 0; maps to 0 
# 0 1 count: 3 label 1; maps to 1 

def test_mdr_transform():
    """Ensure that the MDR 'transform' function maps a new set of feature instances to the desired labels"""
    features = np.array([   [2,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [1,    1],
                            [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDR() 
    mdr.fit(features, classes)
    test_features = np.array([  [2,    2],
                                [1,    1],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [1,    1],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [0,    1],    
                                [1,    0],    
                                [0,    0],    
                                [1,    0],    
                                [0,    0]])

    new_features = mdr.transform(test_features)
    assert np.array_equal(new_features, [[0], [0], [1], [1], [1], [1], [0], [1], [1], [1], [1], [0], [1], [0], [1]])
    

# 2 0 count: 1 label 1; maps to 1 
# 0 0 count: 3 label 0; 6 label 1; maps to 1 *tie_break*
# 1 1 count: 2 label 0; maps to 0 
# 0 1 count: 3 label 1; maps to 1 

def test_mdr_fit_transform():
    """Ensure that the MDR 'fit_transform' function combines both fit and transform, and produces the right predicted labels"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDR() 
    new_features = mdr.fit_transform(features, classes)
    assert np.array_equal(new_features, [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0]])

def test_mdr_score():
    """Ensure that the MDR 'score' function outputs the right default score, as well as the right custom metric if specified"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDRClassifier() 
    mdr.fit(features, classes)
    assert mdr.score(features, classes) == 12. / 15

def test_mdr_custom_score(): 
    """Ensure that the MDR 'score' function outputs the right custom score passed in from the user"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDRClassifier() 
    mdr.fit(features, classes)
    assert mdr.score(features = features, class_labels = classes, scoring_function = accuracy_score) == 12. / 15
    assert mdr.score(features = features, class_labels = classes, scoring_function = zero_one_loss) == 1 - 12. / 15
    assert mdr.score(features = features, class_labels = classes, scoring_function = zero_one_loss, normalize=False) == 15 - 12

def test_mdr_fit_raise_ValueError():
    """Ensure that the MDR 'fit' function raises ValueError when it is not a binary classification (temporary)"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mdr = MDR()
    try:
        mdr.fit(features, classes)
    except ValueError:
        assert True
    else:
        assert False

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    try:
        mdr.fit(features, classes)
    except ValueError:
        assert True
    else:
        assert False

def test_mdr_score_raise_ValueError():
    """Ensure that the MDR 'score' function raises ValueError when 'fit' has not already been called"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    mdr = MDRClassifier()
    
    try:
        mdr.score(features, classes)
    except ValueError:
        assert True
    else:
        assert False

def test_mdr_sklearn_pipeline():
    """Ensure that MDR can be used as a transformer in a scikit-learn pipeline"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    clf = make_pipeline(MDR(), LogisticRegression())
    cv_scores = cross_val_score(clf, features, classes, cv=StratifiedKFold(n_splits=5, shuffle=True))
    assert np.mean(cv_scores) > 0.

def test_mdr_sklearn_pipeline_parallel():
    """Ensure that MDR can be used as a transformer in a parallelized scikit-learn pipeline"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    clf = make_pipeline(MDR(), LogisticRegression())
    cv_scores = cross_val_score(clf, features, classes, cv=StratifiedKFold(n_splits=5, shuffle=True), n_jobs=-1)
    assert np.mean(cv_scores) > 0.


"""
    Continuous MDR tests
"""


def test_continuous_mdr_init():
    """Ensure that the ContinuousMDR instantiator stores the ContinuousMDR variables properly"""

    cmdr_obj = ContinuousMDR() 

    assert cmdr_obj.tie_break == 1
    assert cmdr_obj.default_label == 0
    assert cmdr_obj.overall_mean_trait_value == 0.

    cmdr_obj2 = ContinuousMDR(tie_break=1, default_label=2)

    assert cmdr_obj2.tie_break == 1 
    assert cmdr_obj2.default_label == 2

def test_continuous_mdr_fit():
    """Ensure that the ContinuousMDR 'fit' function constructs the right matrix to count each class, as well as the right map from feature instances to labels"""
    features = np.array([   [2,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [1,    1],
                            [1,    1]])

    targets = np.array([0.9, 0.8, 0.9, 0.85, 0.75, 0.95, 0.9, 1.0, 0.75, 0.8, 0.1, 0.2, 0.3, 0.25, 0.05])

    cmdr = ContinuousMDR() 
    cmdr.fit(features, targets)

    assert len(cmdr.mdr_matrix_values) == 4
    assert len(cmdr.feature_map) == 4

    assert cmdr.mdr_matrix_values[(0, 0)] == [0.8, 0.85, 0.75, 0.95, 1.0, 0.75, 0.1, 0.2, 0.3]
    assert cmdr.mdr_matrix_values[(0, 1)] == [0.9, 0.9, 0.8]
    assert cmdr.mdr_matrix_values[(1, 1)] == [0.25, 0.05]
    assert cmdr.mdr_matrix_values[(2, 0)] == [0.9]

    assert cmdr.feature_map[(0, 0)] == 1
    assert cmdr.feature_map[(0, 1)] == 1
    assert cmdr.feature_map[(1, 1)] == 0
    assert cmdr.feature_map[(2, 0)] == 1

def test_continuous_mdr_transform():
    """Ensure that the ContinuousMDR 'transform' function maps a new set of feature instances to the desired labels"""
    features = np.array([   [2,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    1],
                            [0,    0],
                            [0,    0],
                            [0,    0],
                            [1,    1],
                            [1,    1]])

    targets = np.array([0.9, 0.8, 0.9, 0.85, 0.75, 0.95, 0.9, 1.0, 0.75, 0.8, 0.1, 0.2, 0.3, 0.25, 0.05])

    cmdr = ContinuousMDR() 
    cmdr.fit(features, targets)
    test_features = np.array([  [2,    2],
                                [1,    1],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [1,    1],    
                                [0,    0],    
                                [0,    0],    
                                [0,    0],    
                                [0,    1],    
                                [1,    0],    
                                [0,    0],    
                                [1,    0],    
                                [0,    0]])
    
    expected_outputs = [[0],
                        [0],
                        [1],
                        [1],
                        [1],
                        [1],
                        [0],
                        [1],
                        [1],
                        [1],
                        [1],
                        [0],
                        [1],
                        [0],
                        [1]]

    new_features = cmdr.transform(test_features)
    print(new_features)
    assert np.array_equal(new_features, expected_outputs)

def test_continuous_mdr_fit_transform():
    """Ensure that the ContinuousMDR 'fit_transform' function combines both fit and transform, and produces the right predicted labels"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    targets = np.array([0.9, 0.8, 0.9, 0.85, 0.75, 0.95, 0.9, 1.0, 0.75, 0.8, 0.1, 0.2, 0.3, 0.25, 0.05])
    expected_outputs = [[1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [1],
                        [0],
                        [0]]

    cmdr = ContinuousMDR() 
    new_features = cmdr.fit_transform(features, targets)
    assert np.array_equal(new_features, expected_outputs)

def test_continuous_mdr_score():
    """Ensure that the ContinuousMDR 'score' function outputs the right default score"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    targets = np.array([0.9, 0.8, 0.9, 0.85, 0.75, 0.95, 0.9, 1.0, 0.75, 0.8, 0.1, 0.2, 0.3, 0.25, 0.05])

    cmdr = ContinuousMDR() 
    cmdr.fit(features, targets)
    assert round(cmdr.score(features, targets), 3) == 2.514

def test_continuous_mdr_score_raise_ValueError():
    """Ensure that the ContinuousMDR 'score' function raises ValueError when 'fit' has not already been called"""
    features = np.array([[2,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    1],
                         [0,    0],
                         [0,    0],
                         [0,    0],
                         [1,    1],
                         [1,    1]])

    classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    cmdr = ContinuousMDR()
    
    try:
        cmdr.score(features, classes)
    except ValueError:
        assert True
    else:
        assert False
