#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()
with open('requirements_itk.txt') as requirements_file:
    requirements_itk = requirements_file.read().splitlines()
with open('requirements_dev.txt') as requirements_file:
    requirements_dev = requirements_file.read().splitlines()

setup(name='midatasets',
      version='0.11.0',
      description='Medical Image Dataset tools',
      author='F. K.',
      keywords='medical ',
      license='Apache License 2.0',
      packages=find_packages(include=['midatasets', 'midatasets.*']),
      long_description=readme,
      classifiers=['Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: Python :: 3.4'],
      install_requires=requirements,
      extras_require={
          'all': requirements + requirements_itk,
          'pymongo': requirements + ['pymongo'],
          'dev': requirements + requirements_dev}
      )
