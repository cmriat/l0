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

"""Tests for data processing module."""

import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from upath import UPath

sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from data_processing.utils import validate_datasetdict, validate_lance_dataset
from data_processing.downloader import DOWNLOADER_DICT
from data_processing.dataset_merger import DatasetMerger
from data_processing.quality_assessor import DataAssessor

from l0.utils import get_num_processes


class DummyDataAssessor(DataAssessor):
    """Dummy Data Assessor for testing purposes."""

    @staticmethod
    def process_batch(samples: list[dict]) -> dict:
        """Dummy data assessor that adds an 'assessed' key."""
        for sample in samples:
            # Simulate processing by adding a dummy key
            sample["assessed"] = True
        return samples


class TestDownloadAndAssess:
    """Test class for downloading and assessing datasets."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_temp_dirs(self):
        """Class-scoped fixture to create shared temporary directories and clean up after all tests."""
        # Create temporary directories
        cache_dir = UPath(tempfile.mkdtemp())
        save_dir = UPath(tempfile.mkdtemp())
        merge_dir = UPath(tempfile.mkdtemp())
        assessed_data_dir = UPath(tempfile.mkdtemp())

        # Yield directories to be shared across all tests
        yield {
            "cache_dir": cache_dir,
            "save_dir": save_dir,
            "merge_dir": merge_dir,
            "assessed_data_dir": assessed_data_dir,
        }

        # Cleanup after all tests in the class are complete
        for dir_path in [cache_dir, save_dir, assessed_data_dir]:
            shutil.rmtree(dir_path)

    def test_download_simpleqa_dataset(self, setup_temp_dirs):
        """Tests downloading the SimpleQA dataset using shared temporary directories."""
        cache_dir = setup_temp_dirs["cache_dir"]
        save_dir = setup_temp_dirs["save_dir"]

        assert "simple_qa" in DOWNLOADER_DICT, "simple_qa downloader not found in DOWNLOADER_DICT"

        downloader = DOWNLOADER_DICT["simple_qa"](
            cache_dir=cache_dir,
            save_dir=save_dir,
            num_proc=get_num_processes(4),
        )

        downloader.download()
        assert validate_datasetdict(save_dir / "simple_qa")

    def test_merge_datasets(self, setup_temp_dirs):
        """Tests merging datasets using DatasetMerger with shared temporary directories."""
        save_dir = setup_temp_dirs["save_dir"]
        merge_dir = setup_temp_dirs["merge_dir"]

        dataset_merger = DatasetMerger(
            modified_dataset_dir=save_dir,
            merged_dataset_dir=merge_dir,
            max_sample_per_split=10,
        )

        dataset_merger.merge_datasets()
        assert validate_lance_dataset(merge_dir)

    @patch("atexit.register")
    def test_assess_dataset_with_dummy_assessor(self, mock_atexit_register, setup_temp_dirs):
        """Tests assessing a dataset using DummyDataAssessor and shared temporary directories."""
        mock_atexit_register.return_value = None
        merged_dataset = setup_temp_dirs["merge_dir"]
        assessed_data_dir = setup_temp_dirs["assessed_data_dir"]

        dummy_assessor = DummyDataAssessor(
            merged_dataset=merged_dataset,
            assessed_dataset_dir=assessed_data_dir,
            save_interval=5,
            num_proc=get_num_processes(4),
            batch_size=4,
        )

        dummy_assessor.assess()
        assert validate_lance_dataset(assessed_data_dir)
