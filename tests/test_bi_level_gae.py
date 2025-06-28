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

"""Test for bi-level GAE advantage and return computation."""

import numpy as np
import torch
import pytest
from verl import DataProto
from omegaconf import DictConfig

from l0.verl_adapter.advantage_calculation import compute_bi_level_advantage


@pytest.mark.parametrize(
    "token_level_adv_estimator, step_level_adv_estimator, expected_return, expected_advantage",
    [
        ("trivial", "gae", 1.5, 1.2430),
        ("gae", "gae", 1.5, 1.9014),
        ("trivial", "reinforce_plus_plus", 1.5, 1.3754),
        ("gae", "reinforce_plus_plus", 1.5, 1.9014),
    ],
)
def test_compute_bi_level_advantage_return(
    token_level_adv_estimator, step_level_adv_estimator, expected_return, expected_advantage
):
    batch_size = 4
    response_length = 4
    max_prompt_length = 1

    token_level_rewards = torch.tensor([[0, 0, 0, 0.5], [0, 0, 0, 1], [0, 0, 0, 0.2], [0, 0, 0, 1]])
    step_level_rewards = torch.tensor([0.5, 1, 0.2, 1])

    values = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    step_level_values = torch.tensor([1, 2, 3, 4])

    attention_mask = torch.ones((batch_size, max_prompt_length + response_length), dtype=torch.int32)
    response_mask = torch.ones((batch_size, response_length))
    responses = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [5, 6, 7, 8]])
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_level_rewards,
            "token_level_values": values,
            "step_level_values": step_level_values,
            "step_level_rewards": step_level_rewards,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "max_prompt_length": torch.tensor([max_prompt_length] * batch_size),
            "responses": responses,
        },
        non_tensors={
            "uuids": np.array(["uuid1_traj1", "uuid1_traj1", "uuid2_traj2", "uuid2_traj2"], dtype=object),
            "step_indexes": np.array([0, 1, 0, 1], dtype=int),
        },
    )
    config_dict = DictConfig(
        {
            "step_gamma": 1.0,
            "token_gamma": 1.0,
            "step_lam": 1.0,
            "token_lam": 1.0,
            "token_level_adv_estimator": token_level_adv_estimator,
            "step_level_adv_estimator": step_level_adv_estimator,
        }
    )

    compute_bi_level_advantage(data=data, config=config_dict)
    advantages = data.batch["advantages"]
    returns = data.batch["returns"]
    assert advantages.shape == (batch_size, response_length)
    assert returns.shape == (batch_size, response_length)
    assert advantages[0, 0].item() == pytest.approx(expected_advantage, rel=1e-4)
    assert returns[0, 0].item() == pytest.approx(expected_return, rel=1e-4)
