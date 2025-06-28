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

"""Base dataset downloader."""

from abc import ABC, abstractmethod

import datasets
from upath import UPath
from data_processing.utils import validate_datasetdict

from l0.utils import get_num_processes


class DatasetDownloader(ABC):
    """Base class for dataset downloading and format processing."""

    def __init__(
        self, cache_dir: UPath, save_dir: UPath, dataset_name: str, dataset_id_or_path: str, num_proc: int = 8
    ):
        self.cache_dir = cache_dir
        self.save_dir = save_dir
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        self.dataset_name = dataset_name
        self.dataset_id_or_path = dataset_id_or_path
        self.cache_dataset_dir = self.cache_dir / self.dataset_name
        self.modified_dataset_dir = self.save_dir / self.dataset_name
        self.num_proc = get_num_processes(num_proc)

    def load_dataset(self, subfolder=None):
        return datasets.load_dataset(
            self.dataset_id_or_path,
            name=subfolder,
            cache_dir=self.cache_dataset_dir,
            trust_remote_code=True,
            num_proc=self.num_proc,
        )

    @abstractmethod
    def transform_sample(self, example): ...

    @abstractmethod
    def has_answer(self, example): ...

    def process_dataset(self, dataset):
        """Process the dataset."""
        modified_dataset = datasets.DatasetDict()

        # Process each split in the dataset
        for split in dataset:
            print(f"Processing {split} split...")

            # First filter examples with empty answers
            filtered_dataset = dataset[split].filter(self.has_answer, desc=f"Filtering {split}", num_proc=self.num_proc)

            # Check if filtered dataset is empty
            if len(filtered_dataset) == 0:
                print(f"Warning: All examples in {split} were filtered out.")
                # Create a simple empty dataset with just question and answer fields
                empty_data = {"question": [], "answer": []}
                # Create features for the simplified schema
                features = datasets.Features({"question": datasets.Value("string"), "answer": datasets.Value("string")})
                modified_dataset[split] = datasets.Dataset.from_dict(empty_data, features=features)
            else:
                # Apply the transformation function to all examples in the split
                modified_dataset[split] = filtered_dataset.map(
                    self.transform_sample,
                    remove_columns=filtered_dataset.column_names,
                    desc=f"Processing {split}",
                    num_proc=self.num_proc,
                )

        return modified_dataset

    def save_dataset(self, modified_dataset, subfolder=None):
        """Save the processed dataset to disk."""
        # Create directory for the dataset if it doesn't exist
        save_path = self.modified_dataset_dir
        if subfolder:
            save_path = self.modified_dataset_dir / subfolder

        save_path.mkdir(parents=True, exist_ok=True)

        # Save the entire DatasetDict
        modified_dataset.save_to_disk(save_path)
        print(f"Modified dataset saved to {save_path}")

        # Load and print a sample to verify
        loaded_dataset = datasets.load_from_disk(save_path)
        print(loaded_dataset)
        if "train" in loaded_dataset:
            print(loaded_dataset["train"][0])
        elif "test" in loaded_dataset:
            print(loaded_dataset["test"][0])

    def download(self, subfolder=None) -> list[datasets.DatasetDict]:
        """Main downloading method with checkpoint mechanism."""
        # Ensure dataset_name is set
        if not self.dataset_name:
            raise ValueError("dataset_name must be set")

        # Create directory for modified dataset if it doesn't exist
        if not self.modified_dataset_dir.exists():
            self.modified_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Check if dataset is already processed
        actual_modified_dataset_dir = self.modified_dataset_dir / subfolder if subfolder else self.modified_dataset_dir
        if validate_datasetdict(actual_modified_dataset_dir):
            print(f"Dataset {self.dataset_name} is already processed. Skipping.")
            return [datasets.load_from_disk(actual_modified_dataset_dir)]

        dataset = self.load_dataset(subfolder=subfolder)

        modified_dataset = self.process_dataset(dataset)

        self.save_dataset(modified_dataset, subfolder=subfolder)

        return [modified_dataset]
