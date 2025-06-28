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
"""Configuration for NB Agent worker."""

import os
from dataclasses import field

from upath import UPath
from pydantic import field_validator
from omegaconf import MISSING
from nbagent.models import SGLSamplingParams
from config_dataclass import config_dataclass


def ensure_v_path(v: str | UPath, create_if_not_exists: bool = True) -> UPath:
    if isinstance(v, str):
        v = UPath(v)
    elif not isinstance(v, UPath):
        raise ValueError("Must be a string or UPath instance")
    if not v.exists() and create_if_not_exists:
        v.mkdir(parents=True, exist_ok=True)
    return v


@config_dataclass
class OpenAIServerModelConfig:
    """Configuration for OpenAI server model."""

    model_id: str = "Qwen/Qwen3-0.6B"
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    base_url: str | None = os.getenv("OPENAI_API_BASE")
    server_mode: str = "sglang"


@config_dataclass
class NBAgentConfig:
    """Configuration for NBAgent."""

    uuid: str = MISSING
    max_tokens: int = 25600
    max_response_length: int = 3192
    agent_workspace: str = "./agent_workspace"
    traj_save_storage_path: str | None = None
    virtual_kernel_spec_path: str | None = None
    trim_memory_mode: str = "hard"
    sampling_params: SGLSamplingParams = field(default_factory=SGLSamplingParams)
    agent_mode: str = "llm"
    tool_ability: str = "default"

    @field_validator("agent_workspace", mode="plain")
    @classmethod
    def ensure_workspace_dir(cls, v):
        ensure_v_path(v, create_if_not_exists=True)

    @field_validator("virtual_kernel_spec_path", mode="plain")
    @classmethod
    def ensure_path(cls, v):
        if v is None:
            return v
        return ensure_v_path(v, create_if_not_exists=True)

    @field_validator("traj_save_storage_path", mode="plain")
    @classmethod
    def ensure_path_and_exists(cls, v):
        if v is None:
            return v
        return ensure_v_path(v, create_if_not_exists=True)


@config_dataclass
class TaskRunConfig:
    """Configuration for task run."""

    task: str = MISSING
    max_steps: int = 10
    file_path: str | None = None

    @field_validator("task")
    @classmethod
    def ensure_task_is_a_valid_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Task must be a string.")

        if not v.strip():
            raise ValueError("Task string cannot be empty.")

        return v


@config_dataclass
class NBAgentWorkerConfig:
    """Configuration for NB Agent worker."""

    uuid: str = MISSING
    openai_server_model: OpenAIServerModelConfig = field(default_factory=OpenAIServerModelConfig)
    nb_agent: NBAgentConfig = field(default_factory=NBAgentConfig)
    task_run: TaskRunConfig = field(default_factory=TaskRunConfig)
    log_path: str | None = None

    @field_validator("log_path", mode="plain")
    @classmethod
    def ensure_path_and_exists(cls, v):
        if v is None:
            return v
        return ensure_v_path(v, create_if_not_exists=True)
