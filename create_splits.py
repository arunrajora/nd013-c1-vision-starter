from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    files = glob.glob(os.path.join(data_dir, "*.tfrecord"))

    train_path = os.path.join(data_dir, "train/")
    test_path = os.path.join(data_dir, "test/")
    valid_path = os.path.join(data_dir, "valid/")

    print("Creating train, validation and test directories")
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(valid_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)

    train_indices, valid_test_indices = train_test_split(files, test_size=0.4)
    valid_indices, test_indices = train_test_split(valid_test_indices, test_size=0.5)

    paths = [train_path, valid_path, test_path]
    data_indices = [train_indices, valid_indices, test_indices]

    for path, indices in zip(paths, data_indices):
        print(f"Moving files to {path}.")
        for index in indices:
            Path(index).rename(os.path.join(path, Path(index).name))
        print(f"Added {len(indices)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument("--data_dir", required=True, help="data directory")
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.data_dir)