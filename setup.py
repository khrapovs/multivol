#!/usr/bin/env python

from setuptools import setup, find_packages


with open('README.rst') as file:
    long_description = file.read()

setup(name='multivol',
      version='1.0',
      description=('Multivariate Densities'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='MIT',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/multivol',
      packages=find_packages(),
      package_dir={'deco': './deco'},
      keywords=['volatility', 'multivariate', 'covariance', 'dynamics',
                'estimation', 'simulation'],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
      )
