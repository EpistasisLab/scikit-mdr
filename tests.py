"""
    Unit tests for MDR.
"""

from mdr import MDR 

import pandas as pd
import numpy as np
from collections import Counter
import random
import warnings
import inspect

from sklearn.cross_validation import train_test_split

def test_init():
    """Ensure that the TPOT instantiator stores the TPOT variables properly"""

    mdr_obj = MDR()

    assert mdr_obj.tie_break == 0
    assert mdr_obj.default_label == 0
    assert mdr_obj.class_fraction == 0.
   
def test_fit():
	input_data = pd.read_csv("~/Documents/Models_EDM_1/mod_Models.txt_EDM-1_01.txt", sep='\t')
	training_indices, testing_indices = train_test_split(input_data.index,
                                                         stratify=input_data['Class'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=70)

	training_features = input_data.loc[training_indices].drop('Class', axis=1).values
	training_classes = input_data.loc[training_indices, 'Class'].values

	testing_features = input_data.loc[testing_indices].drop('Class', axis=1).values
	testing_classes = input_data.loc[testing_indices, 'Class'].values
	mdr = MDR() 
	mdr.fit(training_features, training_classes)
	if len(np.unique(training_classes)) != 2: 
		#assert a valueError 
	else:
		assert len(mdr.unique_labels) == 2
	np.array_equal(result['guess'].values, dtc.predict(testing_features))
	assert np.array_equal(mdr.class_count_matrix.keys(), 
	assert mdr.feature_map 


print(mdr.score(testing_features, testing_classes))

