#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs

from setuptools import setup, find_packages

setup(
    name='odis',
    version='0.1.0',
    description='simple models for redis',
    long_description=codecs.open('README', "r", "utf-8").read(),
    author='Simon Klee',
    author_email='simon@simonklee.org',
    url='http://github.com/simonz05/odis',
    license='BSD',
    keywords="redis",
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    zip_safe=True,
    test_suite="nose.collector",
    tests_require=['nose'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
    ],
)
