#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('mdr/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='scikit-MDR',
    version=package_version,
    author='Randal S. Olson and Tuan Nguyen',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/EpistasisLab/scikit-mdr',
    license='License :: OSI Approved :: MIT License',
    description=('Multifactor Dimensionality Reduction (MDR)'),
    long_description='''
A sklearn-compatible Python implementation of Multifactor Dimensionality Reduction (MDR) for feature construction.

Contact
=============
If you have any questions or comments about scikit-MDR, please feel free to contact us via:

E-mail: rso@randalolson.com

or Twitter: https://twitter.com/randal_olson

This project is hosted at https://github.com/EpistasisLab/scikit-mdr
''',
    zip_safe=True,
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['bioinformatics', 'GWAS', 'feature construction', 'single nucleotide polymorphisms', 'epistasis', 'dimesionality reduction', 'scikit-learn', 'machine learning'],
)
