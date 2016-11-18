[![Build Status](https://travis-ci.org/EpistasisLab/scikit-mdr.svg?branch=master)](https://travis-ci.org/EpistasisLab/scikit-mdr)
[![Code Health](https://landscape.io/github/EpistasisLab/scikit-mdr/master/landscape.svg?style=flat)](https://landscape.io/github/EpistasisLab/scikit-mdr/master)
[![Coverage Status](https://coveralls.io/repos/github/EpistasisLab/scikit-mdr/badge.svg?branch=master)](https://coveralls.io/github/EpistasisLab/scikit-mdr?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/scikit-mdr.svg)](https://badge.fury.io/py/scikit-mdr)

# MDR

[![Join the chat at https://gitter.im/EpistasisLab/scikit-mdr](https://badges.gitter.im/EpistasisLab/scikit-mdr.svg)](https://gitter.im/EpistasisLab/scikit-mdr?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A scikit-learn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction. **This project is still under active development** and we encourage you to check back on this repository regularly for updates.

MDR is an effective feature construction algorithm that is capable of modeling epistatic interactions and capturing complex patterns in data sets.

MDR currently only works with categorical features and supports both binary classification and regression problems. We are working on expanding the algorithm to provide more convenience features.

## License

Please see the [repository license](https://github.com/EpistasisLab/scikit-mdr/blob/master/LICENSE) for the licensing and usage information for the MDR package.

Generally, we have licensed the MDR package to make it as widely usable as possible.

## Installation

MDR is built on top of the following existing Python packages:

* NumPy

* SciPy

* scikit-learn

All of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```
conda install numpy scipy scikit-learn
```

Once the prerequisites are installed, you should be able to install MDR with a `pip` command:

```
pip install scikit-mdr
```

Please [file a new issue](https://github.com/EpistasisLab/scikit-mdr/issues/new) if you run into installation problems.

## Examples

MDR has been coded with a scikit-learn-like interface to be easy to use. The typical `fit`, `transform`, and `fit_transform` methods are available for every algorithm. For example, MDR can be used to construct a new feature composed from two existing features:

```python
from mdr import MDR
import pandas as pd

genetic_data = pd.read_csv('https://raw.githubusercontent.com/EpistasisLab/scikit-mdr/master/data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz', sep='\t', compression='gzip')

features = genetic_data.drop('class', axis=1).values
labels = genetic_data['class'].values

my_mdr = MDR()
my_mdr.fit_transform(features, labels)
>>>array([[1],
>>>       [1],
>>>       [1],
>>>       ..., 
>>>       [0],
>>>       [0],
>>>       [0]])
```

You can also estimate the accuracy of the predictions from the constructed feature using the `score` function:

```python
from mdr import MDR
import pandas as pd

genetic_data = pd.read_csv('https://raw.githubusercontent.com/EpistasisLab/scikit-mdr/master/data/GAMETES_Epistasis_2-Way_20atts_0.4H_EDM-1_1.csv.gz', sep='\t', compression='gzip')

features = genetic_data.drop('class', axis=1).values
labels = genetic_data['class'].values

my_mdr = MDR()
my_mdr.fit(features, labels)
my_mdr.score(features, labels)
>>>0.998125
```

If you want to use MDR for regression problems, use `ContinuousMDR`:

```python
from mdr import ContinuousMDR
from sklearn.datasets import load_boston

data = load_boston()
features, targets = data.data, data.target

my_cmdr = ContinuousMDR()
my_cmdr.fit(features, targets)
my_cmdr.score(features, targets)
>>>24.310512404173707
```

## Contributing to MDR

We welcome you to [check the existing issues](https://github.com/EpistasisLab/scikit-mdr/issues/) for bugs or enhancements to work on. If you have an idea for an extension to the MDR package, please [file a new issue](https://github.com/EpistasisLab/scikit-mdr/issues/new) so we can discuss it.

## Having problems or have questions about MDR?

Please [check the existing open and closed issues](https://github.com/EpistasisLab/scikit-mdr/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, [file a new issue](https://github.com/EpistasisLab/scikit-mdr/issues/new) on this repository so we can review your issue.

## Citing MDR

If you use this software in a publication, please consider citing it. You can cite the repository directly with the following DOI:

[blank for now]

## Support for MDR

The MDR package was developed in the [Computational Genetics Lab](http://epistasis.org) with funding from the [NIH](http://www.nih.gov). We're incredibly grateful for their support during the development of this project.
