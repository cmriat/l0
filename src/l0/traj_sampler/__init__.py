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
"""Trajectory sampler."""

from config_dataclass import ConfigDataclass

from .sampler import TrajectorySampler
from .nb_agent_sampler import NBAgentSampler, NBAgentWorkerConfig, get_run_nb_agent_worker_cmd

__all__ = [
    "TrajectorySampler",
    "NBAgentSampler",
    "NBAgentWorkerConfig",
    "get_run_nb_agent_worker_cmd",
]


def get_trajectory_sampler(config: ConfigDataclass) -> TrajectorySampler:
    if config.traj_sampler.agent_type == "nb_agent":
        return NBAgentSampler(config.traj_sampler)
    else:
        raise ValueError(f"Invalid agent type: {config.traj_sampler.agent_type}")
