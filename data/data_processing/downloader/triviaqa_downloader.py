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

"""Trivial QA dataset downloader."""

import datasets
from upath import UPath

from .base_downloader import DatasetDownloader


class TriviaQADownloader(DatasetDownloader):
    """Processor for TriviaQA dataset."""

    def __init__(self, cache_dir: UPath, save_dir: UPath, num_proc: int = 8):
        super().__init__(cache_dir, save_dir, "trivia_qa", "mandarjoshi/trivia_qa", num_proc=num_proc)
        self.config_list = [
            "rc",
            "rc.nocontext",
            "rc.web",
            "rc.web.nocontext",
            "rc.wikipedia",
            "rc.wikipedia.nocontext",
            "unfiltered",
            "unfiltered.nocontext",
        ]

    def transform_sample(self, example):
        return {"question": example["question"], "answer": example["answer"]["value"]}

    def download(self) -> list[datasets.DatasetDict]:
        """Process all configurations of TriviaQA."""
        modified_datasets = []
        for config_name in self.config_list:
            print(f"Processing TriviaQA with config: {config_name}")

            modified_datasets.extend(super().download(subfolder=config_name))

        return modified_datasets

    def has_answer(self, example):
        return len(example["answer"]["value"]) > 0 and example["answer"]["value"] != "<unk>"
