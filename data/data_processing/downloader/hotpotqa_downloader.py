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

"""HotpotQA dataset downloader."""

from upath import UPath

from .base_downloader import DatasetDownloader


class HotpotQADownloader(DatasetDownloader):
    """Processor for HotpotQA dataset."""

    def __init__(self, cache_dir: UPath, save_dir: UPath, num_proc: int = 8):
        super().__init__(cache_dir, save_dir, "hotpot_qa", "hotpotqa/hotpot_qa", num_proc=num_proc)
        self.config_list = ["distractor", "fullwiki"]

    def transform_sample(self, example):
        return {"question": example["question"], "answer": example["answer"]}

    def download(self):
        """Process all configurations of HotpotQA."""
        modified_datasets = []
        for config_name in self.config_list:
            print(f"Processing HotpotQA with config: {config_name}")

            modified_datasets.extend(super().download(subfolder=config_name))

        return modified_datasets

    def has_answer(self, example):
        # Check if answer exists and is not None before getting its length
        return example.get("answer") is not None and len(example["answer"]) > 0
