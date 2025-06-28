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

"""Test cases for the reward functions using pytest."""

from types import SimpleNamespace

import pytest

import l0.verl_adapter.reward.nb_agent_reward as rf


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Hello, World!", "hello world"),
        ("This is a TEST.", "this is a test"),
        ("  Extra   spaces  ", "extra spaces"),
        ("NoPunctuation", "nopunctuation"),
        ("", ""),
        ("123 numbers should be kept.", "123 numbers should be kept"),
        ("!@#$%^&*()", ""),
    ],
)
def test_normalize_text(input_text, expected):
    """Test the _normalize_text function for various inputs."""
    assert rf._normalize_text(input_text) == expected


@pytest.mark.parametrize(
    "prediction, ground_truth, expected_f1",
    [
        ("the cat sat on the mat", "the cat sat on the mat", 1.0),
        ("the cat sat on the mat", "a dog sat on the rug", 0.5),
        ("hello world", "world hello", 1.0),
        ("apple", "banana", 0.0),
        ("", "", 1.0),
        ("a b c", "", 0.0),
        ("", "a b c", 0.0),
        ("repeated words words", "repeated words", 0.8),
    ],
)
def test_compute_f1_score(prediction, ground_truth, expected_f1):
    """Test the _compute_f1_score function."""
    assert rf._compute_f1_score(prediction, ground_truth) == pytest.approx(expected_f1)


@pytest.mark.parametrize(
    "prediction, ground_truth, expected_em",
    [
        ("The CAT sat on the mat.", "the cat sat on the mat", 1.0),
        ("hello world", "world hello", 0.0),  # Order matters for EM after normalization
        ("apple", "Apple", 1.0),
        ("apple", "banana", 0.0),
        ("", "", 1.0),
    ],
)
def test_compute_exact_match(prediction, ground_truth, expected_em):
    """Test the _compute_exact_match function."""
    assert rf._compute_exact_match(prediction, ground_truth) == expected_em


def test_qa_reward_fn():
    """Test the qa_reward_fn, decoupled from hardcoded MIN/MAX values."""
    ground_truth = "The capital of France is Paris"

    # Perfect match should yield the maximum possible reward.
    answers1 = ["The capital of France is Paris"]
    assert rf.qa_reward_fn(ground_truth, answers1)[0]["score"] == [rf.QA_REWARD_MAX]

    # Partial match (high F1, no EM)
    answers2 = ["The capital of France is always Paris"]
    # Calculate raw score first with default weights (f1=0.4, em=0.5, has_answer=0.1)
    raw_f1 = rf._compute_f1_score(answers2[0], ground_truth)  # ~0.923
    raw_em = rf._compute_exact_match(answers2[0], ground_truth)  # 0.0
    raw_has_answer = 1.0  # Has answer
    raw_reward = 0.4 * raw_f1 + 0.5 * raw_em + 0.1 * raw_has_answer
    # Calculate expected reward based on the constants
    expected_reward = rf.QA_REWARD_MIN + (rf.QA_REWARD_MAX - rf.QA_REWARD_MIN) * raw_reward
    assert rf.qa_reward_fn(ground_truth, answers2)[0]["score"] == pytest.approx([expected_reward])

    # No match should still get has_answer reward
    answers3 = ["London"]
    raw_f1_3 = rf._compute_f1_score(answers3[0], ground_truth)  # 0.0
    raw_em_3 = rf._compute_exact_match(answers3[0], ground_truth)  # 0.0
    raw_has_answer_3 = 1.0  # Has answer
    raw_reward_3 = 0.4 * raw_f1_3 + 0.5 * raw_em_3 + 0.1 * raw_has_answer_3
    expected_reward_3 = rf.QA_REWARD_MIN + (rf.QA_REWARD_MAX - rf.QA_REWARD_MIN) * raw_reward_3
    assert rf.qa_reward_fn(ground_truth, answers3)[0]["score"] == pytest.approx([expected_reward_3])

    # Mix of answers including None
    answers4 = ["The capital of France is Paris", None, "No match"]
    expected_mixed = [rf.QA_REWARD_MAX, rf.QA_REWARD_MIN, expected_reward_3]
    assert rf.qa_reward_fn(ground_truth, answers4)[0]["score"] == pytest.approx(expected_mixed)

    # Empty ground truth should be a neutral reward (0.0) as per function logic
    assert rf.qa_reward_fn("", ["Any answer"])[0]["score"] == [(rf.QA_REWARD_MAX + rf.QA_REWARD_MIN) / 2]

    # Custom weights and range test
    rewards = rf.qa_reward_fn(
        ground_truth,
        ["The capital of France is Paris"],
        f1_weight=0.2,
        em_weight=0.7,
        has_answer_weight=0.1,
        max_reward=10.0,
        min_reward=1.0,
    )[0]["score"]
    assert rewards == pytest.approx([10.0])


@pytest.mark.parametrize(
    "response, raw_expected_reward",
    [
        ("<think>This is a well-thought-out plan.</think><code>print('hello')</code>", 1.0),
        ("<code>print('hello')</code>", 0.0),
        ("<think>This is a well-thought-out plan.</think>", 0.0),
        ("<think>  </think><code>print('hello')</code>", 0.0),
        ("<think>This is a well-thought-out plan.</think><code> </code>", 0.0),
        ("<think>1</think><think>2</think><code>c</code>", 0.0),
        ("<think>t</think><code>c1</code><code>c2</code>", 0.0),
    ],
)
def test_format_reward_fn(response, raw_expected_reward):
    """Test the format_reward_fn, decoupled from hardcoded MIN/MAX values."""
    scale_factor = 1  # Testing a single response
    expected_reward = (
        rf.FORMAT_REWARD_MIN + (rf.FORMAT_REWARD_MAX - rf.FORMAT_REWARD_MIN) * raw_expected_reward
    ) / scale_factor
    rewards = rf.format_reward_fn([response])[0]["score"]
    assert rewards == [expected_reward]


def test_format_reward_fn_scaling():
    """Test the scaling factor in format_reward_fn, decoupled from constants."""
    responses = [
        "<think>Valid plan 1.</think><code>print(1)</code>",  # Raw Reward 1.0
        "<think>Invalid plan.</think>",  # Raw Reward 0.0
    ]
    scale_factor = len(responses)

    # Calculate expected rewards based on constants and scaling
    expected_reward_1 = (rf.FORMAT_REWARD_MIN + (rf.FORMAT_REWARD_MAX - rf.FORMAT_REWARD_MIN) * 1.0) / scale_factor
    expected_reward_2 = (rf.FORMAT_REWARD_MIN + (rf.FORMAT_REWARD_MAX - rf.FORMAT_REWARD_MIN) * 0.0) / scale_factor

    rewards = rf.format_reward_fn(responses)[0]["score"]
    assert rewards == [expected_reward_1, expected_reward_2]


@pytest.mark.parametrize(
    "execution_results, raw_expected_reward",
    [
        ([{"output_type": "execute_result", "data": "1"}, {"output_type": "execute_result", "data": "2"}], 1.0),
        ([{"output_type": "execute_result", "data": "1"}, {"output_type": "error", "ename": "ValueError"}], 0.5),
        ([{"output_type": "error", "ename": "TypeError"}, {"output_type": "error", "ename": "NameError"}], 0.0),
        ([], 0.0),
        (None, 0.0),
    ],
)
def test_code_execution_reward_fn(execution_results, raw_expected_reward):
    """Test code_execution_reward_fn, decoupled from hardcoded MIN/MAX values."""
    expected_reward = (
        rf.CODE_EXECUTION_REWARD_MIN
        + (rf.CODE_EXECUTION_REWARD_MAX - rf.CODE_EXECUTION_REWARD_MIN) * raw_expected_reward
    )
    rewards = rf.code_execution_reward_fn([execution_results])[0]["score"]
    assert rewards == [expected_reward]


def create_mock_data_trajectory(ground_truth, full_responses, final_answers, code_execution_results):
    """Helper to create mock data for combine_reward_fn tests."""
    num_steps = len(full_responses)
    traj_ids = list(range(num_steps))
    trajectory = {"index_in_data": traj_ids, "ground_truth": [ground_truth] * num_steps}
    data = SimpleNamespace(non_tensor_batch={})
    data.non_tensor_batch["nb_steps"] = [
        SimpleNamespace(action=SimpleNamespace(full_response=resp), observations=obs)
        for resp, obs in zip(full_responses, code_execution_results, strict=True)
    ]
    data.non_tensor_batch["meta_info"] = [{"final_answer": ans} for ans in final_answers]
    return data, trajectory


def test_combine_reward_fn_default_weights():
    """Test combine_reward_fn with default weights, decoupled from constants."""
    data, trajectory = create_mock_data_trajectory(
        ground_truth="The answer is 42",
        full_responses=["<think>The user wants the answer... That's 42.</think><code>print(42)</code>"],
        final_answers=["The answer is 42"],
        code_execution_results=[[{"output_type": "execute_result", "data": "42"}]],
    )

    weights = {"qa": 0.8, "length": 0, "format": 0.1, "code_execution": 0.1}

    # Calculate expected rewards for each component using constants
    # QA reward: perfect match + has_answer = 1.0 (all components max)
    qa_r = rf.QA_REWARD_MAX  # Perfect match
    format_r = rf.FORMAT_REWARD_MAX  # Perfect format, scale factor is 1
    code_r = rf.CODE_EXECUTION_REWARD_MAX  # No errors
    length_r = 0  # Weight is 0, value doesn't matter

    expected_combined = (
        weights["qa"] * qa_r
        + weights["length"] * length_r
        + weights["format"] * format_r
        + weights["code_execution"] * code_r
    )

    combined_rewards = rf.combine_str_reward_fn(data, trajectory)[0]["score"]
    assert combined_rewards == pytest.approx([expected_combined])


def test_combine_reward_fn_custom_weights():
    """Test combine_reward_fn with custom weights and a failing case, decoupled from constants."""
    data, trajectory = create_mock_data_trajectory(
        ground_truth="The answer is 42",
        full_responses=["<think>I am not sure.</think><code>import error</code>"],
        final_answers=["I don't know"],
        code_execution_results=[[{"output_type": "error", "ename": "ModuleNotFoundError"}]],
    )

    # Use only existing reward components (no length since weight is 0 by default)
    reward_weights = {"qa": 0.6, "format": 0.2, "code_execution": 0.2}

    # Calculate expected component rewards using constants
    # For QA: "I don't know" vs "The answer is 42" - should get has_answer reward only
    raw_f1 = rf._compute_f1_score("I don't know", "The answer is 42")  # 0.0
    raw_em = rf._compute_exact_match("I don't know", "The answer is 42")  # 0.0
    raw_has_answer = 1.0  # Has answer
    raw_qa_reward = 0.4 * raw_f1 + 0.5 * raw_em + 0.1 * raw_has_answer
    qa_r = rf.QA_REWARD_MIN + (rf.QA_REWARD_MAX - rf.QA_REWARD_MIN) * raw_qa_reward

    format_r = rf.FORMAT_REWARD_MAX  # Has both think and code tags
    code_r = rf.CODE_EXECUTION_REWARD_MIN  # Has error

    expected_combined = (
        reward_weights["qa"] * qa_r + reward_weights["format"] * format_r + reward_weights["code_execution"] * code_r
    )

    combined_rewards = rf.combine_str_reward_fn(data, trajectory, reward_weights=reward_weights)[0]["score"]
    assert combined_rewards == pytest.approx([expected_combined])
