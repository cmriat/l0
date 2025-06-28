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
"""Process and filter assessed datasets."""

import argparse

import dotenv
from upath import UPath
from data_processing.filter import DataFilter


def get_parser():
    parser = argparse.ArgumentParser(description="Filter assessed datasets based on quality ratings")
    parser.add_argument(
        "--assessed_data_dir",
        type=str,
        required=True,
        help="Directory containing the assessed datasets",
    )
    parser.add_argument(
        "--filtered_data_dir",
        type=str,
        required=True,
        help="Directory to save filtered datasets",
    )
    parser.add_argument(
        "--objectivity_threshold",
        type=float,
        default=1,
        help="Minimum objectivity rating to include (0-1)",
    )
    parser.add_argument(
        "--temporal_stability_threshold",
        type=float,
        default=1,
        help="Minimum temporal stability rating to include (0-1)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser


def main():
    dotenv.load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    # Convert paths to UPath objects
    assessed_data_dir = UPath(args.assessed_data_dir)
    filtered_data_dir = UPath(args.filtered_data_dir)

    # Initialize and run the filter
    data_filter = DataFilter(
        assessed_dataset_dir=assessed_data_dir,
        filtered_dataset_dir=filtered_data_dir,
        objectivity_threshold=args.objectivity_threshold,
        temporal_stability_threshold=args.temporal_stability_threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    data_filter.filter()


if __name__ == "__main__":
    main()
