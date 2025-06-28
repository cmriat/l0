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

"""Tests for MultistepRewardManager."""

import random
from unittest.mock import MagicMock

import torch
import pytest
from verl import DataProto

from l0.verl_adapter.reward.multistep_reward_manager import MultistepRewardManager

SAMPLE_SIZE = 5
STEP_INDEXES = random.sample(range(SAMPLE_SIZE), SAMPLE_SIZE)


class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            # transform tensor to list of integers
            ids = ids.flatten().cpu().tolist()
            # Convert list of integers to string
            ids = "".join(map(str, ids))
            return ids


@pytest.fixture
def test_data():
    """Fixture for test data."""
    return DataProto.from_dict(
        non_tensors={
            "uuids": ["uuid_0"] * SAMPLE_SIZE,
            "reward_model": [{"ground_truth": "test_truth"}] * SAMPLE_SIZE,
            "data_source": ["test_data_source"] * SAMPLE_SIZE,
            "question": ["test_question"] * SAMPLE_SIZE,
            "step_indexes": STEP_INDEXES,
        },
        tensors={
            "responses": torch.tensor([[i, i, i] for i in range(SAMPLE_SIZE)]),
            "attention_mask": torch.ones((SAMPLE_SIZE, 6), dtype=torch.int64),
            "max_prompt_length": torch.tensor([3] * SAMPLE_SIZE, dtype=torch.int64),
        },
    )


@pytest.fixture
def test_trajectories():
    """Fixture for test trajectories."""
    return [
        {
            "actions": [str(STEP_INDEXES.index(i)) * 3 for i in range(SAMPLE_SIZE)],
            "ground_truth": ["test_truth"] * SAMPLE_SIZE,
            "index_in_data": [STEP_INDEXES.index(i) for i in range(SAMPLE_SIZE)],
            "valid_lengths": [3] * SAMPLE_SIZE,
            "data_source": ["test_data_source"] * SAMPLE_SIZE,
            "question": ["test_question"] * SAMPLE_SIZE,
        }
    ]


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def reward_manager(mock_tokenizer):
    return MultistepRewardManager(
        tokenizer=mock_tokenizer, num_examine=2, compute_score=None, reward_fn_key="data_source"
    )


def test_init_with_compute_score():
    """Test initialization with compute_score parameter."""
    mock_tokenizer = MockTokenizer()
    with pytest.raises(NotImplementedError):
        MultistepRewardManager(
            tokenizer=mock_tokenizer, num_examine=2, compute_score=lambda x: x, reward_fn_key="data_source"
        )


def test_collect_trajectory_data(reward_manager, test_data, test_trajectories):
    """Test _group_trajectories method."""
    trajectories = reward_manager._group_trajectories(test_data)

    assert len(trajectories) == 1
    trajectory = trajectories[0]

    # step_indexes should be removed after sorting
    assert "step_indexes" not in trajectory

    for key in trajectory.keys():
        assert trajectory[key] == test_trajectories[0][key]


def test_compute_rewards(reward_manager, test_data, test_trajectories):
    """Test _compute_rewards method."""
    mock_compute_score = MagicMock(return_value=(list(range(SAMPLE_SIZE)), {}))
    reward_manager.compute_score = mock_compute_score

    _, token_level_reward_tensor, _, _, _ = reward_manager._compute_rewards(test_data, test_trajectories)
    assert isinstance(token_level_reward_tensor, torch.Tensor)
    # will test reordering of reward_tensor
    assert token_level_reward_tensor.shape == (SAMPLE_SIZE, 3)
    assert token_level_reward_tensor[0, 0] == 0
    assert token_level_reward_tensor[0, 2] == STEP_INDEXES[0]
    assert token_level_reward_tensor[2, 1] == 0
    assert token_level_reward_tensor[1, 2] == STEP_INDEXES[1]
