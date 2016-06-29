"""
Tuan Nguyen & Randal Olson 
Summer 2016 
Unit tests for MDR
"""

from mdr import MDR 
import numpy as np
import random
import warnings
import inspect
from sklearn.cross_validation import train_test_split

#full dataset for testing 
# 2	0	1
# 0	0	1
# 0	1	1
# 0	0	1
# 0	0	1
# 0	0	1
# 0	1	1
# 0	0	1
# 0	0	1
# 0	1	1
# 0	0	0
# 0	0	0
# 0	0	0
# 1	1	0
# 1	1	0

# 2 0 count: 1 label 1; maps to 1 
# 0 0 count: 3 label 0; 6 label 1; maps to 0 *tie_break*
# 1 1 count: 2 label 0; maps to 0 
# 0 1 count: 3 label 1; maps to 1 

def test_init():
    """Ensure that the MDR instantiator stores the MDR variables properly"""

    mdr_obj = MDR() #change this or create a second test 

    assert mdr_obj.tie_break == 0
    assert mdr_obj.default_label == 0
    assert mdr_obj.class_fraction == 0.

    mdr_obj2 = MDR(tie_break = 1, default_label = 2)

    assert mdr_obj2.tie_break == 1 
    assert mdr_obj2.default_label == 2


def test_fit():

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
	assert mdr.feature_map[(0,0)] == 0
	assert mdr.feature_map[(1,1)] == 0
	assert mdr.feature_map[(0,1)] == 1

def test_transform():
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
	for row_i in range(test_features.shape[0]):
		assert new_features[row_i] == mdr.feature_map[tuple(test_features[row_i])]
	assert new_features[0] == mdr.default_label
	assert new_features[13] == mdr.default_label

def test_fit_transform():
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
	new_features = mdr.fit_transform(features, classes)
	for row_i in range(new_features.shape[0]):
		assert new_features[row_i] == mdr.feature_map[tuple(features[row_i])]
	assert new_features[0] == 1
	assert new_features[13] == 0

test_init()
test_fit()
test_transform()
test_fit_transform()

# have not written a test for scoring method 
