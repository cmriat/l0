# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for data processing."""

import logging

import lance
from upath import UPath
from datasets import load_from_disk

log = logging.getLogger("backoff.retry")


def _log_backoff(details):
    # on_exception → 用 'exception'
    sample = details["args"][0]
    exc = details["exception"]
    wait = details["wait"]
    print(f"[retry #{details['tries']}] {exc!r} — sleep {wait:.2f}s")
    print(sample)


def _log_giveup(details):
    print(f"[give-up] after {details['tries']} tries: {details['exception']!r}")


def validate_dataset(dataset_dir: UPath) -> bool:
    """Validate a dataset dictionary.

    Args:
        dataset: A dictionary representing a dataset.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """
    if not dataset_dir.exists():
        return False

    # Check for dataset_dict.json which is essential for HF datasets
    if not (dataset_dir / "dataset_info.json").exists():
        return False

    try:
        # Attempt to load the dataset
        load_from_disk(dataset_dir)
    except Exception as e:
        return False

    return True


def validate_datasetdict(dataset_dir: UPath):
    """Validate the modified dataset directory.

    Args:
        dataset_dir: Directory containing a datasetdict.
    """
    if not dataset_dir.exists():
        return False

    # Check for dataset_dict.json which is essential for HF datasets
    if not (dataset_dir / "dataset_dict.json").exists():
        return False

    try:
        # Attempt to load the dataset
        load_from_disk(dataset_dir)
    except Exception as e:
        return False

    return True


def find_all_possible_modified_datasets(base_dir: UPath) -> list[UPath]:
    # check if it's a directory and has a file 'dataset_dict.json'
    candidates_dirs = base_dir.glob("**/**")
    # check if it's a directory and has a file 'dataset_dict.json'
    valid_datasets = []
    for candidate_dir in candidates_dirs:
        if candidate_dir.is_dir() and (candidate_dir / "dataset_dict.json").exists():
            valid_datasets.append(candidate_dir)
    return valid_datasets


def validate_lance_dataset(dataset_dir: UPath):
    """Validate if the directory contains a Lance dataset.

    Args:
        dataset_dir: Directory suspected to contain a Lance dataset.
    """
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return False

    try:
        lance.dataset(dataset_dir)
    except Exception:
        return False

    return True
