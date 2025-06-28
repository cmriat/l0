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

"""Filter for processing assessed datasets."""

import random
from typing import Tuple

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from upath import UPath


class DataFilter:
    def __init__(
        self,
        assessed_dataset_dir: UPath,
        filtered_dataset_dir: UPath,
        objectivity_threshold: float = 0.7,
        temporal_stability_threshold: float = 0.7,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize the data filter.

        Args:
            assessed_dataset_dir: Directory containing the assessed dataset
            filtered_dataset_dir: Directory to save filtered datasets
            objectivity_threshold: Minimum objectivity rating to include (0-1)
            temporal_stability_threshold: Minimum temporal stability rating to include (0-1)
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            seed: Random seed for reproducibility
        """
        self.assessed_dataset_dir = assessed_dataset_dir
        self.filtered_dataset_dir = filtered_dataset_dir
        self.objectivity_threshold = objectivity_threshold
        self.temporal_stability_threshold = temporal_stability_threshold
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Set random seed
        random.seed(seed)

    def load_assessed_dataset(self) -> lance.LanceDataset:
        """Load the assessed dataset from disk."""
        return lance.dataset(self.assessed_dataset_dir)

    def filter_samples(self, dataset: lance.LanceDataset) -> pa.Table:
        """Filter samples based on rating thresholds."""
        # Convert to PyArrow table for filtering
        table = dataset.to_table()

        # Create filter masks using pyarrow.compute
        objectivity_mask = pc.greater_equal(table["objectivity rating"], self.objectivity_threshold)
        temporal_stability_mask = pc.greater_equal(
            table["temporal stability rating"], self.temporal_stability_threshold
        )
        assessed_mask = pc.equal(table["assessed"], True)

        # Combine masks
        combined_mask = pc.and_(pc.and_(objectivity_mask, temporal_stability_mask), assessed_mask)

        # Apply filter
        filtered_table = table.filter(combined_mask)
        return filtered_table

    def split_dataset(self, table: pa.Table) -> Tuple[pa.Table, pa.Table, pa.Table]:
        """Split the filtered dataset into train/val/test sets."""
        # Convert to list for shuffling
        data = table.to_pylist()
        random.shuffle(data)

        # Calculate split indices
        n_samples = len(data)
        train_end = int(n_samples * self.train_ratio)
        val_end = train_end + int(n_samples * self.val_ratio)

        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # Convert back to PyArrow tables
        train_table = pa.Table.from_pylist(train_data, schema=table.schema)
        val_table = pa.Table.from_pylist(val_data, schema=table.schema)
        test_table = pa.Table.from_pylist(test_data, schema=table.schema)

        return train_table, val_table, test_table

    def transform_data_format(self, table: pa.Table) -> pa.Table:
        """Transform data into the required format.

        The new format is:
        {
            'question': 'Who received the IEEE Frank Rosenblatt Award in 2010?',
            'answer': 'Michio Sugeno',
            'id': 0,
            'ability': 'qa',
            'reward_model': {
                'ground_truth': 'Michio Sugeno',
                'style': 'rule'
            },
            'extra_info': {
                'answer': 'Michio Sugeno',
                'assessment': {
                    'Objectivity Rating': 1,
                    'Objectivity Rationale': [...],
                    'Temporal Stability Rating': 1,
                    'Temporal Stability Rationale': [...]
                },
                'question': 'Who received the IEEE Frank Rosenblatt Award in 2010?'
            }
        }
        """
        # Convert to list for transformation
        data_list = table.to_pylist()
        transformed_data = []

        for i, item in enumerate(data_list):
            # Create the new format
            transformed_item = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "id": item.get("id", i),
                "ability": "qa",
                "reward_model": {"ground_truth": item.get("answer", ""), "style": "rule"},
                "extra_info": {
                    "answer": item.get("answer", ""),
                    "assessment": {
                        "Objectivity Rating": item.get("objectivity rating", -1),
                        "Objectivity Rationale": item.get("objectivity rationale", []),
                        "Temporal Stability Rating": item.get("temporal stability rating", -1),
                        "Temporal Stability Rationale": item.get("temporal stability rationale", []),
                    },
                    "question": item.get("question", ""),
                },
            }
            transformed_data.append(transformed_item)

        # Convert back to PyArrow table
        transformed_table = pa.Table.from_pylist(transformed_data)
        return transformed_table

    def save_splits(self, train_table: pa.Table, val_table: pa.Table, test_table: pa.Table):
        """Save the split datasets to disk as parquet files."""
        # Create output directory
        self.filtered_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Transform data to the required format
        transformed_train_table = self.transform_data_format(train_table)
        transformed_val_table = self.transform_data_format(val_table)
        transformed_test_table = self.transform_data_format(test_table)

        # Save each split as parquet file
        pq.write_table(transformed_train_table, self.filtered_dataset_dir / "train.parquet")
        pq.write_table(transformed_val_table, self.filtered_dataset_dir / "validation.parquet")
        pq.write_table(transformed_test_table, self.filtered_dataset_dir / "test.parquet")

        # Print statistics
        print(f"Saved filtered datasets:")
        print(f"Train: {len(train_table)} samples -> {self.filtered_dataset_dir}/train.parquet")
        print(f"Validation: {len(val_table)} samples -> {self.filtered_dataset_dir}/validation.parquet")
        print(f"Test: {len(test_table)} samples -> {self.filtered_dataset_dir}/test.parquet")

    def filter(self):
        """Main filtering pipeline."""
        # Load dataset
        dataset = self.load_assessed_dataset()

        # Filter samples
        filtered_table = self.filter_samples(dataset)
        print(f"Filtered {filtered_table.num_rows} samples from {dataset.count_rows()} total samples")

        # Split dataset
        train_table, val_table, test_table = self.split_dataset(filtered_table)

        # Save splits
        self.save_splits(train_table, val_table, test_table)
