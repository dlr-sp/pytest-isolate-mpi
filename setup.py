#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('CHANGES.rst', encoding='utf-8') as history_file:
    changes = history_file.read()

REQUIREMENTS = [
    'pytest >= 5',
    'mpi4py',
    # add additional project dependencies here
]

DEV_REQUIREMENTS = REQUIREMENTS + [
    'check-manifest',
    'Sphinx<7',
    'sphinx-rtd-theme<2',
    'docutils<0.19',
    'pylint',
    'pytest-cov',
    'wheel',
    'numpy',
    'black'
    # add additional development dependencies here
]

setup(
    name='pytest_isolate_mpi',
    author='German Aerospace Center (DLR e.V.)',
    version='0.1.0',
    description="pytest-isolate-mpi allows for MPI-parallel tests being executed in a segfault and MPI_Abort safe manner",
    long_description=readme + '\n\n' + changes,
    python_requires='>=3.8',
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        'dev': DEV_REQUIREMENTS,
    },
    entry_points={
        'console_scripts': [
            'pytest_isolate_mpi=pytest_isolate_mpi.__main__:main',
        ],
    },
)
