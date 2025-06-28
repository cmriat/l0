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

"""Comprehensive reward functions for Agent training."""

import re
from typing import Any
from collections import Counter

# Reward value constants
QA_REWARD_MIN = 0.0
QA_REWARD_MAX = 1.0
LENGTH_REWARD_MIN = 0.0
LENGTH_REWARD_MAX = 1.0
FORMAT_REWARD_MIN = 0.0
FORMAT_REWARD_MAX = 1.0
CODE_EXECUTION_REWARD_MIN = 0.0
CODE_EXECUTION_REWARD_MAX = 1.0


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and removing punctuation."""
    if not text:
        return ""
    text = text.lower()
    # Keep spaces and alphanumeric, remove others
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())  # Remove extra whitespace
    return text


def _compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # Calculate intersection
    intersection = sum((pred_counter & gt_counter).values())

    # Calculate precision and recall
    precision = intersection / len(pred_tokens) if pred_tokens else 0.0
    recall = intersection / len(gt_tokens) if gt_tokens else 0.0

    # Calculate F1
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth."""
    pred_normalized = _normalize_text(prediction)
    gt_normalized = _normalize_text(ground_truth)
    return 1.0 if pred_normalized == gt_normalized else 0.0


def qa_reward_fn(
    ground_truth: str,
    final_answers: list[str | None],
    f1_weight: float = 0.4,
    em_weight: float = 0.5,
    has_answer_weight: float = 0.1,
    max_reward: float = QA_REWARD_MAX,
    min_reward: float = QA_REWARD_MIN,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """QA Reward function that computes F1 and EM scores.

    Args:
        ground_truth: The ground truth answer
        final_answers: list of generated answers, None values are skipped
        f1_weight: Weight for F1 score (default: 0.4)
        em_weight: Weight for EM score (default: 0.5)
        has_answer_weight: Weight for has_answer score (default: 0.1)
        max_reward: Maximum reward value (default: QA_REWARD_MAX)
        min_reward: Minimum reward value (default: QA_REWARD_MIN)

    Returns:
        A list of reward scores for each generated answer
    """
    if not ground_truth:
        return {"score": [(min_reward + max_reward) / 2] * len(final_answers)}, {
            "f1_score": 0.0,
            "em_score": 0.0,
            "has_answer_score": 0.0,
        }

    rewards = []
    for answer in final_answers:
        if answer is None or not isinstance(answer, str):
            # Skip None values, assign neutral reward
            rewards.append(min_reward)
            continue

        has_answer_score = 1.0
        f1_score = _compute_f1_score(answer, ground_truth)
        em_score = _compute_exact_match(answer, ground_truth)

        # Compute raw reward (0-1 range)
        raw_reward = f1_weight * f1_score + em_weight * em_score + has_answer_weight * has_answer_score

        # Scale to [min_reward, max_reward] range
        scaled_reward = min_reward + (max_reward - min_reward) * raw_reward
        rewards.append(scaled_reward)

    # Compute detailed metrics for the last answer (for logging)
    last_answer = final_answers[-1] if final_answers else None
    if last_answer and isinstance(last_answer, str):
        last_f1_score = _compute_f1_score(last_answer, ground_truth)
        last_em_score = _compute_exact_match(last_answer, ground_truth)
        last_has_answer_score = 1.0
    else:
        last_f1_score = 0.0
        last_em_score = 0.0
        last_has_answer_score = 0.0

    return {"score": rewards}, {
        "f1_score": last_f1_score,
        "em_score": last_em_score,
        "core_score": last_em_score,
        "has_answer_score": last_has_answer_score,
    }


# TODO: add step level reward
def length_reward_fn(
    full_responses: list[str],
    target_len_words: int,
    max_reward: float = LENGTH_REWARD_MAX,
    min_reward: float = LENGTH_REWARD_MIN,
    penalty_if_missing: bool = True,
    too_short_penalty_factor: float = 0.5,
    too_long_penalty_factor: float = 0.5,
    tolerance_factor: float = 0.2,
) -> tuple[dict[str, list[float]], dict]:
    """Length Reward function that checks if full responses meet length requirements.

    Args:
        full_responses: List of full response strings
        target_len_words: Target length in words
        max_reward: Maximum reward value (default: LENGTH_REWARD_MAX)
        min_reward: Minimum reward value (default: LENGTH_REWARD_MIN)
        penalty_if_missing: Whether to penalize missing content
        too_short_penalty_factor: Penalty factor for too short content
        too_long_penalty_factor: Penalty factor for too long content
        tolerance_factor: Tolerance factor for target length (e.g., +/- 20%)

    Returns:
        A list of reward scores for each full response
    """
    rewards = []

    for text_content in full_responses:
        if not text_content:
            reward = min_reward if penalty_if_missing else (min_reward + max_reward) / 2
            rewards.append(reward)
            continue

        num_words = len(text_content.split())

        if num_words == 0 and penalty_if_missing:
            rewards.append(min_reward)
            continue

        if target_len_words <= 0:  # Avoid division by zero if target length is invalid
            rewards.append((min_reward + max_reward) / 2)
            continue

        lower_bound = target_len_words * (1 - tolerance_factor)
        upper_bound = target_len_words * (1 + tolerance_factor)

        if lower_bound <= num_words <= upper_bound:
            reward = max_reward
        elif num_words < lower_bound:
            shortage_ratio = num_words / lower_bound
            # Penalty calculation similar to dummy_reward.py
            penalty = (max_reward - min_reward) * too_short_penalty_factor * (1.0 - shortage_ratio)
            reward = max(min_reward, max_reward - penalty)
        else:  # num_words > upper_bound
            # Penalize for being too long, similar logic
            excess_ratio = (num_words - upper_bound) / upper_bound  # How much percentage wise it's over
            penalty = (max_reward - min_reward) * too_long_penalty_factor * min(1.0, excess_ratio)  # Cap penalty effect
            reward = max(min_reward, max_reward - penalty)

        rewards.append(reward)

    return {
        "score": rewards,
        "avg_word_count": [
            sum(len(text.split()) for text in full_responses if text) / len([text for text in full_responses if text])
            if any(full_responses)
            else 0
        ]
        * len(rewards),
        "target_word_count": [target_len_words] * len(rewards),
    }, {}


def _extract_think_content(text: str) -> list[str]:
    """Extract content from <think>...</think> tags."""
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, text, re.DOTALL)
    return matches


def _extract_code_content(text: str) -> list[str]:
    """Extract content from <code>...</code> tags."""
    code_pattern = r"<code>(.*?)</code>"
    matches = re.findall(code_pattern, text, re.DOTALL)
    return matches


def format_reward_fn(
    full_responses: list[str], max_reward: float = FORMAT_REWARD_MAX, min_reward: float = FORMAT_REWARD_MIN
) -> tuple[dict[str, list[float]], dict]:
    """Format Reward function that checks if code and think items have proper format.

    Args:
        codes: List of code strings
        thinks: List of think strings
        max_reward: Maximum reward value (default: FORMAT_REWARD_MAX)
        min_reward: Minimum reward value (default: FORMAT_REWARD_MIN)

    Returns:
        A list of reward scores for each code/think pair
    """
    rewards = []
    # reward will be scaled by step nums to avoid reward hacking
    scale_factor = len(full_responses)

    for full_response in full_responses:
        # Check think content using extraction function
        think_contents = _extract_think_content(full_response)
        code_contents = _extract_code_content(full_response)

        if len(think_contents) > 1 or len(code_contents) > 1 or not think_contents or not code_contents:
            rewards.append(min_reward / scale_factor)
            continue

        think_content = think_contents[0]
        code_content = code_contents[0]
        think_has_content = len(think_content.strip()) > 10  # Require some meaningful content
        code_has_content = len(code_content.strip()) > 5  # Require some code content

        raw_reward = int(think_has_content and code_has_content)

        # Scale to [min_reward, max_reward] range
        scaled_reward = min_reward + (max_reward - min_reward) * raw_reward
        rewards.append(scaled_reward / scale_factor)

    return {"score": rewards}, {"traj_format_score": sum(rewards)}


def code_execution_reward_fn(
    code_execution_results: list[list[dict] | None],
    max_reward: float = CODE_EXECUTION_REWARD_MAX,
    min_reward: float = CODE_EXECUTION_REWARD_MIN,
) -> tuple[dict[str, list[float]], dict]:
    """Code Execution Reward function that checks code execution results from code cells.

    Args:
        code_cells: list of lists of code cell dictionaries containing execution information
        max_reward: Maximum reward value (default: CODE_EXECUTION_REWARD_MAX)
        min_reward: Minimum reward value (default: CODE_EXECUTION_REWARD_MIN)

    Returns:
        A list of reward scores for each code cells group
    """
    rewards = []
    scale_factor = len(code_execution_results)

    for code_execution_result in code_execution_results:
        if code_execution_result is None:
            rewards.append(min_reward)
            continue

        output_count = len(code_execution_result)
        error_count = sum(
            1 for output in code_execution_result if isinstance(output, dict) and output.get("output_type") == "error"
        )

        raw_reward = (1 - error_count / output_count) if output_count > 0 else 0.0

        # Scale the raw reward to [min_reward, max_reward] range
        scaled_reward = min_reward + (max_reward - min_reward) * raw_reward
        scaled_reward /= scale_factor  # Scale by number of steps
        rewards.append(scaled_reward)

    return {"score": rewards}, {"traj_code_execution_score": sum(rewards)}


def combine_str_reward_fn(
    data,
    trajectory: dict,
    reward_weights: dict[str, float] | None = None,
    qa_config: dict[str, Any] | None = None,
    length_config: dict[str, Any] | None = None,
    format_config: dict[str, Any] | None = None,
    code_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine multiple reward functions with configurable weights.

    Args:
        data: DataProto object containing the input data
        trajectory: dictionary containing trajectory information with 'actions' key
        reward_weights: dictionary with weights for each reward type
            - 'qa': Weight for QA reward
            - 'length': Weight for length reward
            - 'format': Weight for format reward
            - 'code_execution': Weight for code execution reward
        qa_config: Configuration for QA reward function
        length_config: Configuration for length reward function
        format_config: Configuration for format reward function
        code_config: Configuration for code execution reward function (currently unused)

    Returns:
        A dictionary containing the combined reward scores and additional information
    """
    # Set default weights
    if reward_weights is None:
        reward_weights = {"qa": 0.8, "length": 0, "format": 0.1, "code_execution": 0.1}

    # Set default configurations
    if qa_config is None:
        qa_config = {"f1_weight": 0.0, "em_weight": 0.9, "has_answer_weight": 0.1}

    if length_config is None:
        length_config = {
            "target_len_words": 50,
            "penalty_if_missing": True,
            "too_short_penalty_factor": 0.5,
            "too_long_penalty_factor": 0.5,
            "tolerance_factor": 0.2,
        }

    if format_config is None:
        format_config = {}

    if code_config is None:
        code_config = {}

    # Extract data from trajectory
    traj_ids = trajectory["index_in_data"]
    ground_truth = trajectory["ground_truth"][0]
    full_responses = [data.non_tensor_batch["nb_steps"][step_id].action.full_response for step_id in traj_ids]
    final_answers = [data.non_tensor_batch["meta_info"][step_id]["final_answer"] for step_id in traj_ids]
    code_execution_results = []
    for step_id in traj_ids:
        try:
            code_execution_result = data.non_tensor_batch["nb_steps"][step_id].observations
        except AttributeError:
            code_execution_result = None
        code_execution_results.append(code_execution_result)

    # Compute individual rewards
    qa_result, qa_raw_metrics = qa_reward_fn(ground_truth, final_answers, **qa_config)
    length_result, length_raw_metrics = length_reward_fn(full_responses, **length_config)
    format_result, format_raw_metrics = format_reward_fn(full_responses, **format_config)
    code_result, code_raw_metrics = code_execution_reward_fn(code_execution_results, **code_config)

    # Extract scores from results
    qa_rewards = qa_result["score"]
    length_rewards = length_result["score"]
    format_rewards = format_result["score"]
    code_rewards = code_result["score"]

    raw_metrics = {
        "length_score": length_rewards,
        "format_score": format_rewards,
        "code_execution_score": code_rewards,
    }
    raw_metrics.update(qa_raw_metrics)
    raw_metrics.update(length_raw_metrics)
    raw_metrics.update(format_raw_metrics)
    raw_metrics.update(code_raw_metrics)

    # Ensure all reward lists have the same length
    assert len(qa_rewards) == len(length_rewards) == len(format_rewards) == len(code_rewards)

    # Combine rewards with weights
    combined_rewards = []
    for qa_r, length_r, format_r, code_r in zip(qa_rewards, length_rewards, format_rewards, code_rewards, strict=False):
        combined_score = (
            reward_weights.get("qa", 0) * qa_r
            + reward_weights.get("length", 0) * length_r
            + reward_weights.get("format", 0) * format_r
            + reward_weights.get("code_execution", 0) * code_r
        )
        combined_rewards.append(combined_score)

    result = {
        "score": combined_rewards,
    }
    result.update({f"qa_{k}": v for k, v in qa_result.items() if k != "score"})
    result.update({f"length_{k}": v for k, v in length_result.items() if k != "score"})
    result.update({f"format_{k}": v for k, v in format_result.items() if k != "score"})
    result.update({f"code_{k}": v for k, v in code_result.items() if k != "score"})

    return result, raw_metrics
