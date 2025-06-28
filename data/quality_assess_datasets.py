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
"""Assess dataset quality."""

import argparse

import lance
import dotenv
from upath import UPath
from data_processing.quality_assessor import DataAssessor


def get_parser():
    parser = argparse.ArgumentParser(description="Assess quality of merged datasets")
    parser.add_argument(
        "--merged_data_dir", type=str, required=True, help="Directory containing the merged LanceDataset"
    )
    parser.add_argument("--assessed_data_dir", type=str, required=True, help="Directory to save assessed datasets")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Number of examples to save in each shard (default: 1000)",
    )
    parser.add_argument("--num_proc", type=int, default=48, help="Number of processes for inference")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples to process in each batch")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to assess (for testing)"
    )
    return parser


def main():
    dotenv.load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    merged_data_dir = UPath(args.merged_data_dir)
    assessed_data_dir = UPath(args.assessed_data_dir)

    # Load the merged LanceDataset
    print(f"Loading merged LanceDataset from {merged_data_dir}...")
    merged_dataset = lance.dataset(str(merged_data_dir))

    # Optionally limit the number of samples for testing
    if args.max_samples is not None:
        print(f"Limiting to {args.max_samples} samples for testing")
        # Create a limited view of the dataset
        limited_table = merged_dataset.to_table().slice(0, args.max_samples)
        merged_dataset = lance.write_dataset(
            limited_table, str(merged_data_dir.parent / f"{merged_data_dir.name}_limited"), mode="overwrite"
        )

    print(f"Loaded dataset with {len(merged_dataset)} samples")

    # Assess datasets using DataAssessor
    print("Assessing dataset quality...")
    data_assessor = DataAssessor(
        merged_dataset=merged_dataset,
        assessed_dataset_dir=assessed_data_dir,
        save_interval=args.save_interval,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
    )
    data_assessor.assess()
    print("Dataset assessment completed!")


if __name__ == "__main__":
    main()
