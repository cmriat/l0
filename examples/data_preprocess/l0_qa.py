# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# This file is adapted from the verl library.
# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates.
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
#
# NOTICE: This file has been modified by China Merchants Research Institute Of Advanced Technology from its original version.
"""l0-qa dataset preprocessing script."""

import os
import argparse

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="cmriat/l0-clearqa")

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.data_source)

    for split, dset in dataset.items():
        dset.to_parquet(os.path.join(args.local_dir, f"{split}.parquet"))
