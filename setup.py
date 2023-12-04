#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()


def parse_requirements(path):
    with open(path) as requirements_file:
        requirements_base = requirements_file.read().splitlines()

    requirements = [
        f"{r.split('#egg=')[-1]}@{r}" for r in requirements_base if r.startswith("git+")
    ]
    requirements += [r for r in requirements_base if not r.startswith("git+")]
    return requirements


requirements = parse_requirements("requirements.txt")
requirements_itk = parse_requirements("requirements_itk.txt")
requirements_dev = parse_requirements("requirements_dev.txt")


setup(
    name="midatasets",
    version="0.25.2",
    description="Medical Image Dataset tools",
    author="F. K.",
    keywords="medical ",
    license="Apache License 2.0",
    packages=find_packages(include=["midatasets", "midatasets.*"]),
    long_description=readme,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.4",
    ],
    install_requires=requirements,
    extras_require={
        "all": requirements + requirements_itk,
        "pymongo": requirements + ["pymongo"],
        "dev": requirements + requirements_dev,
    },
)
