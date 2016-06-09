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

import pandas as pd
import numpy as np 
from __future__ import print_function
from ._version import __version__

class MDR(object):

    """Multifactor Dimensionality Reduction (MDR) for feature construction in machine learning"""

    def __init__(self, tie_break = 0, default_label = 0):
        """Sets up the MDR algorithm

        Parameters
        ----------
        tie_break: type int (default: 0)
            description: specify the default label in case there's a tie in a given set of feature values 
        default_label: type int (default: 0)
            description: specify the default label in case there's no data for a given set of feature values  

        Returns
        -------
        None

        """
        self.tie_break = tie_break
        self.default_label = default_label
        self.fdict = {} 

    def fit(self, features, classes):
        """Constructs the MDR feature map from the provided training data

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
        n = (np.unique(classes)).size # general case 
        for i in range(features.shape[0]):
            feature_instance = tuple(map(tuple, features[i])) 
            if feature_instance not in dict: 
                fdict[feature_instance] = np.zeros((n,), dtype=np.int) #initialize count for a new set of feature values 
            #assuming there are only 2 unique label values - might have to change to generalize
            if classes[i] = 0:
                fdict[feature_instance][0] += 1
            else:
                fdict[feature_instance][1] += 1

        




        

    def transform(self, features):
        """Uses the MDR feature map to construct a new feature from the provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to transform

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        new_feature = np.zeros(shape=(features.shape[0],1))
        for i in range(features.shape[0]):
            feature_instance = tuple(map(tuple, features[i]))
            if feature_instance in fdict:
                counts = fdict[feature_instance]
                if counts[0] > counts[1]:
                    new_feature[i] = 0
                elif counts[1] > counts[0]:
                    new_feature[i] = 1
                else:
                    new_feature[i] = self.tie_break
            else:
                new_feature[i] = self.default_label
            
        return new_feature

    def fit_transform(self, features, classes):
        """Convenience function that fits the provided data then constructs a new feature from the provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        ----------
        array-like: {n_samples}
            Constructed features from the provided feature matrix

        """
        self.fit(features, classes)
        return self.transform(features)

    def score(self, features, classes):
        """Estimates the accuracy of the predictions from the constructed feature

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to be transformed
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        accuracy_score: float
            The estimated accuracy based on the constructed feature

        """
        new_feature = self.fit_transform(features, classes)
        results = (new_feature == classes).all()
        score = np.sum(results)
        accuracy_score = score/classes.size 
        return accuracy_score

def main():
    """Main function that is called when MDR is run on the command line"""
    parser = argparse.ArgumentParser(description='Multifactor Dimensionality Reduction (MDR) for feature construction in machine learning.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to perform MDR on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=0,
                        type=int, help='Random number generator seed for reproducibility. Set this seed if you want your MDR run to be reproducible '
                                       'with the same seed and data set in the future.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information MDR communicates while it is running: 0 = none, 1 = minimal, 2 = all.')

    parser.add_argument('--version', action='version', version='MDR {version}'.format(version=__version__),
                        help='Show MDR\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nMDR settings:')
        for arg in sorted(args.__dict__):
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    training_indices, testing_indices = train_test_split(input_data.index,
                                                         stratify=input_data['class'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE)

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    # Run and evaluate MDR on the training and testing data
    mdr = MDR()
    mdr.fit(training_features, training_classes)
    return mdr.score(testing_features, testing_classes)

if __name__ == '__main__':
    main()
