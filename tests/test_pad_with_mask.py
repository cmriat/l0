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
"""This test is used to test the pad_with_mask and unpad_with_mask functions."""

import numpy as np
import torch
from verl import DataProto

from l0.verl_adapter.ray_trainer import pad_with_mask, unpad_with_mask


def test_pad_unpad():
    # Create a simple DataProto object
    # Create some tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples
    attention_mask = torch.ones(3, 2)  # 3 samples' attention mask

    # Create some non-tensor data
    labels = np.array(["a", "b", "c"], dtype=object)

    # Create DataProto
    data = DataProto.from_dict(
        tensors={"obs": obs, "attention_mask": attention_mask},
        non_tensors={"labels": labels},
        meta_info={"info": "test_info"},
    )

    # Test case 1: Padding when data length is not divisible by world_size
    world_size = 2
    padded_data = pad_with_mask(data, world_size)

    # Verify padded data
    assert len(padded_data) == 4  # should be padded to 4 samples
    assert torch.all(padded_data.batch["pad_mask"][:3] == 1)  # first 3 are original data
    assert torch.all(padded_data.batch["pad_mask"][3:] == 0)  # last 1 is padded data
    print(padded_data.batch["attention_mask"])
    assert padded_data.batch["attention_mask"][3, 0] == 1
    assert padded_data.batch["attention_mask"][3, 1] == 0

    # Test unpadding
    unpadded_data = unpad_with_mask(padded_data, world_size)

    # Verify unpadded data is the same as original data
    assert len(unpadded_data) == 3  # should be restored to original length
    assert torch.all(unpadded_data.batch["obs"] == data.batch["obs"])  # tensor data should be the same
    assert (
        unpadded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]
    ).all()  # non-tensor data should be the same

    # Test case 2: When data length is divisible by world_size
    world_size = 3
    padded_data = pad_with_mask(data, world_size)

    # Verify data is not padded
    assert len(padded_data) == 3  # length should be the same
    assert torch.all(padded_data.batch["pad_mask"] == 1)  # all data should be original data

    # Test unpadding again
    unpadded_data = unpad_with_mask(padded_data, world_size)

    # Verify unpadded data is the same as original data
    assert len(unpadded_data) == 3
    assert torch.all(unpadded_data.batch["obs"] == data.batch["obs"])
    assert (unpadded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]).all()

    # Test case 3: world size 4, data size 5
    world_size = 4
    data_size = 5
    obs = torch.tensor([[i, i + 1] for i in range(data_size)])  # 5 samples
    attention_mask = torch.ones(data_size, 2)  # 5 samples' attention mask
    labels = np.array([str(i) for i in range(data_size)], dtype=object)

    data = DataProto.from_dict(
        tensors={"obs": obs, "attention_mask": attention_mask},
        non_tensors={"labels": labels},
        meta_info={"info": "test_info"},
    )

    padded_data = pad_with_mask(data, world_size)

    # Verify padded data
    assert len(padded_data) == 8  # should be padded to 8 samples (2 buckets of 4)
    assert torch.equal(padded_data.batch["pad_mask"], torch.tensor([1, 1, 1, 0, 1, 0, 1, 0]))

    # Test unpadding
    unpadded_data = unpad_with_mask(padded_data, world_size)

    # Verify unpadded data is the same as original data
    assert len(unpadded_data) == 5  # should be restored to original length
    assert torch.all(unpadded_data.batch["obs"] == data.batch["obs"])  # tensor data should be the same
    assert (
        unpadded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]
    ).all()  # non-tensor data should be the same

    # Test case 4: world size 4, data size 7
    world_size = 4
    data_size = 7
    obs = torch.tensor([[i, i + 1] for i in range(data_size)])  # 7 samples
    attention_mask = torch.ones(data_size, 2)  # 7 samples' attention mask
    labels = np.array([str(i) for i in range(data_size)], dtype=object)

    data = DataProto.from_dict(
        tensors={"obs": obs, "attention_mask": attention_mask},
        non_tensors={"labels": labels},
        meta_info={"info": "test_info"},
    )

    padded_data = pad_with_mask(data, world_size)

    # Verify padded data
    assert len(padded_data) == 8  # should be padded to 8 samples (2 buckets of 4)
    assert torch.all(padded_data.batch["pad_mask"][:7] == 1)  # first 7 are original data
    assert torch.all(padded_data.batch["pad_mask"][7:] == 0)  # last 1 is padded data
    assert torch.all(padded_data.batch["attention_mask"][7:, 1:] == 0)  # padded data's attention mask should be 0

    # Test unpadding
    unpadded_data = unpad_with_mask(padded_data, world_size)

    # Verify unpadded data is the same as original data
    assert len(unpadded_data) == 7  # should be restored to original length
    assert torch.all(unpadded_data.batch["obs"] == data.batch["obs"])  # tensor data should be the same
    assert (
        unpadded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]
    ).all()  # non-tensor data should be the same

    # Test case 5: world size 4, data size 8
    world_size = 4
    data_size = 8
    obs = torch.tensor([[i, i + 1] for i in range(data_size)])  # 8 samples
    attention_mask = torch.ones(data_size, 2)  # 8 samples' attention mask
    labels = np.array([str(i) for i in range(data_size)], dtype=object)

    data = DataProto.from_dict(
        tensors={"obs": obs, "attention_mask": attention_mask},
        non_tensors={"labels": labels},
        meta_info={"info": "test_info"},
    )

    padded_data = pad_with_mask(data, world_size)

    # Verify padded data
    assert len(padded_data) == 8  # should not be padded as it's already divisible by world_size
    assert torch.all(padded_data.batch["pad_mask"] == 1)  # all entries should be original data
    assert torch.all(padded_data.batch["attention_mask"] == 1)  # all attention masks should be 1

    # Test unpadding
    unpadded_data = unpad_with_mask(padded_data, world_size)

    # Verify unpadded data is the same as original data
    assert len(unpadded_data) == 8  # should be the same length
    assert torch.all(unpadded_data.batch["obs"] == data.batch["obs"])  # tensor data should be the same
    assert (
        unpadded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]
    ).all()  # non-tensor data should be the same
