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
"""Utility functions."""

import copy
import socket
import multiprocessing
from typing import Any, Dict, List, Callable

import torch
from verl import DataProto
from verl.utils.torch_functional import masked_var, masked_mean


def get_local_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except Exception as e:
        print(f"Unable to get local IP address: {e}")
        ip_address = "127.0.0.1"
    finally:
        if "s" in locals():
            s.close()
    return ip_address


def get_num_processes(num_processes: int | None = None, reserved_processes: int = 1) -> int:
    default_num_processes = max(1, multiprocessing.cpu_count() - reserved_processes)
    if num_processes is None:
        num_processes = default_num_processes
    num_processes = min(num_processes, default_num_processes)
    return num_processes


def pad_with_mask(data: DataProto, world_size: int) -> DataProto:
    if len(data) % world_size != 0:
        bucket_size = len(data) // world_size + 1
        buckets = []
        data.batch["pad_mask"] = torch.ones(len(data))
        pad_size = world_size - len(data) % world_size
        cur_idx = 0
        for i in range(world_size):
            if i < world_size - pad_size:
                buckets.append(data.select_idxs([j for j in range(cur_idx, cur_idx + bucket_size)]))
                cur_idx += bucket_size
            else:
                pad_data = copy.deepcopy(data[0:1])
                pad_data.batch["pad_mask"].fill_(0)
                if "attention_mask" in pad_data.batch:
                    pad_data.batch["attention_mask"].fill_(0)
                    # to avoid errors in remove padding, also, since loss will be masked by response length
                    # this won't affect the loss
                    pad_data.batch["attention_mask"][0, 0] = 1
                buckets.append(
                    DataProto.concat(
                        [data.select_idxs([j for j in range(cur_idx, cur_idx + bucket_size - 1)]), pad_data]
                    )
                )
                cur_idx += bucket_size - 1
        data_padded = DataProto.concat(buckets)
        return data_padded
    else:
        # For data that's already divisible by world_size, all entries are original
        data.batch["pad_mask"] = torch.ones(len(data))
        return data


def unpad_with_mask(data: DataProto, world_size: int) -> DataProto:
    selected_idxs = []
    bucket_size = len(data) // world_size
    for i in range(world_size):
        if data.batch["pad_mask"][i * bucket_size + bucket_size - 1] == 1:
            selected_idxs.extend([j for j in range(i * bucket_size, i * bucket_size + bucket_size)])
        else:
            selected_idxs.extend([j for j in range(i * bucket_size, i * bucket_size + bucket_size - 1)])

    return data.select_idxs(selected_idxs)


def collect_trajectory_data(
    data: DataProto,
    fields_to_collect: Dict[str, Callable[[Any], Any]],
) -> List[Dict[str, Any]]:
    """Collect data for all trajectories.

    Args:
        data: DataProto object containing the trajectory data
        fields_to_collect: Dictionary mapping field names to functions that extract the field value from a step
        sort_by_step_index: Whether to sort the trajectory steps by their step index

    Returns:
        List of dictionaries containing the collected trajectory data
    """
    # Group data by trajectory identifiers
    trajectories = {}
    for i, step in enumerate(data):
        # Get trajectory identifier
        key = step.non_tensor_batch["uuids"]

        # Initialize or update trajectory data
        if key not in trajectories:
            trajectories[key] = {field_name: [] for field_name in fields_to_collect.keys()}
            trajectories[key]["index_in_data"] = []
            trajectories[key]["valid_lengths"] = []
            trajectories[key]["step_indexes"] = []

        # Collect field values
        for field_name, extract_fn in fields_to_collect.items():
            trajectories[key][field_name].append(extract_fn(step))

        trajectories[key]["index_in_data"].append(i)
        valid_response_length = step.batch["attention_mask"][step.batch["max_prompt_length"] :].sum()
        trajectories[key]["valid_lengths"].append(valid_response_length)
        trajectories[key]["step_indexes"].append(step.non_tensor_batch["step_indexes"])

    trajectories = list(trajectories.values())

    # Sort trajectories by their step indexes if requested
    for traj_index in range(len(trajectories)):
        step_indexes = trajectories[traj_index]["step_indexes"]
        sorted_indices = sorted(range(len(step_indexes)), key=lambda x: step_indexes[x])
        trajectories[traj_index].pop("step_indexes")
        for key, traj_value in trajectories[traj_index].items():
            if isinstance(traj_value, list) and len(traj_value) > 0:
                trajectories[traj_index][key] = [traj_value[i] for i in sorted_indices]

    return trajectories


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    if mask.sum() == 1:
        return values
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened
