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

"""Simple QA dataset downloader."""

from upath import UPath

from .base_downloader import DatasetDownloader


class SimpleQADownloader(DatasetDownloader):
    """Processor for SimpleQA dataset."""

    def __init__(self, cache_dir: UPath, save_dir: UPath, num_proc: int = 8):
        super().__init__(cache_dir, save_dir, "simple_qa", "basicv8vc/SimpleQA", num_proc=num_proc)

    def transform_sample(self, example):
        return {"question": example["problem"], "answer": example["answer"]}

    def has_answer(self, example):
        return len(example["answer"]) > 0
