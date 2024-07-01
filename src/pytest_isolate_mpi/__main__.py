"""Console script for pytest-isolate-mpi."""

import argparse
import sys

import numpy as np

from .core import sum_array


def main():
    """Console script for pytest-isolate-mpi."""
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'), help='path to input data file')
    args = parser.parse_args()
    array = np.loadtxt(args.file)
    print(f'{sum_array(array)=}')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
