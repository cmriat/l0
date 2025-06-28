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
"""Abstract base class for dataset loaders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Sequence

from verl import DataProto
from upath import UPath


class BaseDataset(ABC):
    """Abstract dataset interface providing minimal common functionality."""

    # Default split names; subclasses may override if needed
    SPLIT_NAMES: Tuple[str, ...] = ("train", "dev", "test")

    def __init__(self, data_dir: str | UPath) -> None:
        """Initialize dataset with the given root directory (supports UPath)."""
        self._root = UPath(data_dir)
        self._splits: Dict[str, List] = {}

    @abstractmethod
    def _load_local(self) -> None:
        """Load and validate all dataset splits, populating `self._splits`."""
        ...

    @abstractmethod
    def _load_hf(self) -> None:
        """Load and validate all dataset splits, populating `self._splits` from Huggingface."""
        ...

    @abstractmethod
    def build_batch(
        self,
        *,
        split: str,
        n_samples: int | None = None,
    ) -> Tuple[DataProto, Sequence]:
        """Convert split data into a `DataProto` batch and return labels.

        Args:
            split: Name of the split (`train`, `dev`, or `test`).
            n_samples: Optional limit on number of samples; defaults to all.

        Returns:
            A tuple `(batch, labels)` where `batch` is a DataProto for model input,
            and `labels` is a sequence of ground-truth values.
        """
        ...

    def __len__(self) -> int:
        """Return total number of samples across all splits."""
        return sum(len(samples) for samples in self._splits.values())

    def get_split(self, split: str) -> List:
        """Retrieve a list of samples for the specified split.

        Raises:
            KeyError: If the split name is not recognized.
        """
        if split not in self._splits:
            raise KeyError(f"Unknown split '{split}'. Available splits: {list(self._splits)}")
        return self._splits[split]

    def __iter__(self):
        """Iterate over all samples in order: train, dev, test."""
        for split in self.SPLIT_NAMES:
            for sample in self._splits.get(split, []):
                yield sample
