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

"""Natural Questions dataset downloader."""

import datasets
from upath import UPath

from .base_downloader import DatasetDownloader


class NaturalQuestionsDownloader(DatasetDownloader):
    """Processor for Natural Questions dataset."""

    def __init__(self, cache_dir: UPath, save_dir: UPath, num_proc: int = 8):
        super().__init__(
            cache_dir, save_dir, "natural_questions", "google-research-datasets/natural_questions", num_proc=num_proc
        )

    def has_answer(self, example):
        """Check if an example has a non-empty answer."""
        return len(example["annotations"]["short_answers"][0]["text"]) > 0

    def transform_sample(self, example):
        return {
            "question": example["question"]["text"],
            "answer": ", ".join(example["annotations"]["short_answers"][0]["text"]),
        }

    def process_dataset(self, dataset):
        """Process the dataset with filtering for non-empty answers."""
        # Create a DatasetDict to hold all modified splits
        modified_dataset = datasets.DatasetDict()

        # Process each split in the dataset
        for split in dataset:
            print(f"Processing {split} split...")

            # First filter examples with empty answers
            filtered_dataset = dataset[split].filter(self.has_answer, desc=f"Filtering {split}", num_proc=self.num_proc)

            # Apply the transformation function to all examples in the split
            modified_dataset[split] = filtered_dataset.map(
                self.transform_sample,
                remove_columns=filtered_dataset.column_names,
                desc=f"Processing {split}",
                num_proc=self.num_proc,
            )

        return modified_dataset
