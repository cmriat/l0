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

"""Dataset merger for concatenating modified datasets."""

import lance
from upath import UPath
from datasets import Dataset, load_from_disk, concatenate_datasets

from .utils import find_all_possible_modified_datasets


class DatasetMerger:
    """Class for merging modified datasets into a single LanceDataset."""

    ASSESSMENT_COLUMN_DEFAULTS = {
        "assessed": False,
        "objectivity rating": -2,
        "objectivity rationale": [""],
        "temporal stability rating": -2,
        "temporal stability rationale": [""],
    }

    def __init__(
        self,
        modified_dataset_dir: UPath,
        merged_dataset_dir: UPath,
        max_sample_per_split: int | None = None,
    ):
        """Initialize the DatasetMerger.

        Args:
            modified_dataset_dir: Directory containing the modified datasets
            merged_dataset_dir: Directory to save the merged LanceDataset
            max_sample_per_split: Maximum number of examples per split (optional)
        """
        self.modified_dataset_dir = modified_dataset_dir
        self.merged_dataset_dir = merged_dataset_dir
        self.max_sample_per_split = max_sample_per_split

    def concat_modified_dataset(self) -> Dataset | None:
        """Concatenate all modified datasets into a single HuggingFace Dataset.

        Returns:
            Dataset: Concatenated HuggingFace dataset with assessment columns added
        """
        datasets_list = []

        for dataset_dir in find_all_possible_modified_datasets(self.modified_dataset_dir):
            dataset_name_parts = dataset_dir.relative_to(self.modified_dataset_dir).parts
            dataset_name_str = "/".join(dataset_name_parts)
            dataset = load_from_disk(dataset_dir)

            # Process each split in the dataset
            for split in dataset.keys():
                dataset_split = dataset[split]

                # Apply sample limit if specified
                if self.max_sample_per_split is not None:
                    dataset_split = dataset_split.select(range(min(len(dataset_split), self.max_sample_per_split)))

                # Add metadata columns
                dataset_split = dataset_split.add_column("split", [split] * len(dataset_split))
                dataset_split = dataset_split.add_column("dataset_name", [dataset_name_str] * len(dataset_split))
                datasets_list.append(dataset_split)

        if not datasets_list:
            return None

        # Concatenate all datasets
        concated_dataset = concatenate_datasets(datasets_list)

        # Add assessment columns with default values
        for col_name, default_value in self.ASSESSMENT_COLUMN_DEFAULTS.items():
            concated_dataset = concated_dataset.add_column(col_name, [default_value] * len(concated_dataset))

        # Add unique ID column
        concated_dataset = concated_dataset.add_column("id", [i for i in range(len(concated_dataset))])

        return concated_dataset

    def merge_datasets(self) -> lance.LanceDataset:
        """Merge modified datasets and save as LanceDataset.

        Returns:
            lance.LanceDataset: The merged LanceDataset
        """
        # Check if LanceDataset already exists
        if (
            self.merged_dataset_dir.exists()
            and (self.merged_dataset_dir / "_dataset" / "data" / "manifest.json").exists()
        ):
            print(f"LanceDataset already exists at {self.merged_dataset_dir}")
            return lance.dataset(str(self.merged_dataset_dir))

        # Concatenate HuggingFace datasets first
        hf_dataset = self.concat_modified_dataset()

        if hf_dataset is None:
            raise ValueError("No valid datasets found in the specified directory to concatenate.")

        # Create output directory
        self.merged_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Convert HuggingFace dataset to LanceDataset
        lance_dataset = lance.write_dataset(hf_dataset, str(self.merged_dataset_dir), mode="overwrite")

        print(f"Merged dataset saved to {self.merged_dataset_dir} as LanceDataset")
        print(f"Total samples: {len(lance_dataset)}")

        return lance_dataset

    def get_or_create_merged_dataset(self) -> lance.LanceDataset:
        """Get existing LanceDataset or create a new one by merging.

        Returns:
            lance.LanceDataset: The merged LanceDataset
        """
        return self.merge_datasets()
