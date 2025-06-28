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
"""Merge modified datasets into a single LanceDataset."""

import argparse

import dotenv
from upath import UPath
from data_processing.dataset_merger import DatasetMerger


def get_parser():
    parser = argparse.ArgumentParser(description="Merge all datasets in the modified directory")
    parser.add_argument("--modified_data_base_dir", type=str, required=True, help="Directory containing the datasets")
    parser.add_argument("--merged_data_dir", type=str, required=True, help="Directory to save merged LanceDataset")
    parser.add_argument(
        "--max_sample_per_split",
        nargs="?",
        const=None,
        default=None,
        type=int,
        help="Maximum number of examples to process per split",
    )
    return parser


def merge_datasets(modified_data_base_dir: UPath, merged_data_dir: UPath, max_sample_per_split: int = None):
    """Merge datasets using DatasetMerger.

    Args:
        modified_data_base_dir: Directory containing the modified datasets
        merged_data_dir: Directory to save merged LanceDataset
        max_sample_per_split: Maximum number of examples per split (optional)

    Returns:
        lance.LanceDataset: The merged LanceDataset
    """
    print("Merging modified datasets...")
    dataset_merger = DatasetMerger(
        modified_dataset_dir=modified_data_base_dir,
        merged_dataset_dir=merged_data_dir,
        max_sample_per_split=max_sample_per_split,
    )
    merged_dataset = dataset_merger.get_or_create_merged_dataset()
    print(f"Merged dataset ready with {len(merged_dataset)} samples")
    return merged_dataset


def main():
    dotenv.load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    modified_data_base_dir = UPath(args.modified_data_base_dir)
    merged_data_dir = UPath(args.merged_data_dir)

    merge_datasets(
        modified_data_base_dir=modified_data_base_dir,
        merged_data_dir=merged_data_dir,
        max_sample_per_split=args.max_sample_per_split,
    )
    print("Dataset merging completed!")


if __name__ == "__main__":
    main()
