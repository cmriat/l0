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
"""Dataset class for Question Answering tasks."""

import os
import copy
from collections import defaultdict

import numpy as np
import torch
import datasets
from upath import UPath
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class QADataset(Dataset):
    def __init__(self, data_files: str | list[str], tokenizer: PreTrainedTokenizer, config: DictConfig):
        if not isinstance(data_files, list):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.config = config
        self.tokenizer = tokenizer

        self.cache_dir = UPath(config.get("cache_dir", "~/.cache/verl/qa")).expanduser()

        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        # default not store
        self.serialize_dataset = False
        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.encode(doc[prompt_key])) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        if self.image_key in row_dict:
            raise NotImplementedError("Image data is not supported yet.")
        if self.video_key in row_dict:
            raise NotImplementedError("Video data is not supported yet.")

        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["data_source"] = row_dict["source"]
        row_dict["question"] = "Give me the anwser directly: " + row_dict.get("question", "")

        return row_dict

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._read_files()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "dataframe" not in self.__dict__:
            self._read_files()
