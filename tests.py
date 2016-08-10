"""
Tuan Nguyen & Randal Olson 
Summer 2016 
Unit tests for MDR
"""

from mdr import MDR 
import numpy as np
import unittest
import random
import warnings
import inspect
from sklearn.metrics import accuracy_score, zero_one_loss

def test_init():
    """Ensure that the MDR instantiator stores the MDR variables properly"""

    mdr_obj = MDR() 

    assert mdr_obj.tie_break == 1
    assert mdr_obj.default_label == 0
    assert mdr_obj.class_fraction == 0.

    mdr_obj2 = MDR(tie_break = 1, default_label = 2)

    assert mdr_obj2.tie_break == 1 
    assert mdr_obj2.default_label == 2

def test_fit():
	"""Ensure that the MDR 'fit' method constructs the right matrix to count each class, as well as the right map from feature instances to labels"""
	features = np.array([   [2,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[1,	1],
							[1,	1]])

	classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

	mdr = MDR() 
	mdr.fit(features, classes)

	assert len(mdr.unique_labels) == 2
	assert mdr.class_fraction == 1. / 3.
	assert len(mdr.class_count_matrix) == 4
	assert len(mdr.feature_map) == 4

	assert mdr.class_count_matrix[(2,0)][0] == 0 
	assert mdr.class_count_matrix[(2,0)][1] == 1
	assert mdr.class_count_matrix[(0,0)][0] == 3 
	assert mdr.class_count_matrix[(0,0)][1] == 6
	assert mdr.class_count_matrix[(1,1)][0] == 2 
	assert mdr.class_count_matrix[(1,1)][1] == 0 
	assert mdr.class_count_matrix[(0,1)][0] == 0 
	assert mdr.class_count_matrix[(0,1)][1] == 3 
	assert mdr.class_count_matrix[(2,2)][0] == 0
	assert mdr.class_count_matrix[(2,2)][1] == 0

	assert mdr.feature_map[(2,0)] == 1
	assert mdr.feature_map[(0,0)] == 1
	assert mdr.feature_map[(1,1)] == 0
	assert mdr.feature_map[(0,1)] == 1

# 2 0 count: 1 label 1; maps to 1 
# 0 0 count: 3 label 0; 6 label 1; maps to 1 *tie_break*
# 1 1 count: 2 label 0; maps to 0 
# 0 1 count: 3 label 1; maps to 1 

def test_transform():
	"""Ensure that the MDR 'transform' method maps a new set of feature instances to the desired labels"""
	features = np.array([   [2,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[1,	1],
							[1,	1]])

	classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

	mdr = MDR() 
	mdr.fit(features, classes)
	test_features = np.array([	[2, 2],
								[1,	1],	
								[0,	0],	
								[0,	0],	
								[0,	0],	
								[0,	0],	
								[1,	1],	
								[0,	0],	
								[0,	0],	
								[0,	0],	
								[0,	1],	
								[1,	0],	
								[0,	0],	
								[1,	0],	
								[0,	0]])

	new_features = mdr.transform(test_features)
	assert np.array_equal(new_features, [0,0,1,1,1,1,0,1,1,1,1,0,1,0,1])

# 2 0 count: 1 label 1; maps to 1 
# 0 0 count: 3 label 0; 6 label 1; maps to 1 *tie_break*
# 1 1 count: 2 label 0; maps to 0 
# 0 1 count: 3 label 1; maps to 1 

def test_fit_transform():
	"""Ensure that the MDR 'fit_transform' method combines both fit and transform, and produces the right predicted labels"""
	features = np.array([[2,0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[1,	1],
						[1,	1]])

	classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

	mdr = MDR() 
	new_features = mdr.fit_transform(features, classes)
	assert np.array_equal(new_features, [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0])

def test_score():
	"""Ensure that the MDR 'score' method outputs the right default score, as well as the right custom metric if specified"""
	features = np.array([[2,0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[1,	1],
						[1,	1]])

	classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

	mdr = MDR() 
	mdr.fit(features, classes)
	assert mdr.score(features, classes)	== 12./15

def test_custom_score(): 
	"""Ensure that the MDR 'score' method outputs the right custom score passed in from the user"""
	features = np.array([[2,0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	1],
						[0,	0],
						[0,	0],
						[0,	0],
						[1,	1],
						[1,	1]])

	classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

	mdr = MDR() 
	mdr.fit(features, classes)
	assert mdr.score(features = features, classes = classes, scoring_function = accuracy_score) == 12./15
	assert mdr.score(features = features, classes = classes, scoring_function = zero_one_loss) == 1 - 12./15
	assert mdr.score(features = features, classes = classes, scoring_function = zero_one_loss, normalize=False) == 15 - 12

class Test_fit_raise_ValueError(unittest.TestCase):
	def test_fit_raise_ValueError(self):
		"""Ensure that the MDR 'fit' method raises ValueError when it is not a binary classification (temporary)"""
		features = np.array([[2,0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	1],
							[0,	0],
							[0,	0],
							[0,	0],
							[1,	1],
							[1,	1]])

		classes = np.array([1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

		mdr = MDR()
		#self.assertRaises(ValueError, mdr.score, features, classes)
		self.assertRaises(ValueError, mdr.fit, features, classes)
		classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
		self.assertRaises(ValueError, mdr.fit, features, classes)

