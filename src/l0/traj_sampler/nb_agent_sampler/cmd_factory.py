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
"""run cmd for nb_agent."""

import os
import sys
import json
import shlex
import subprocess
from copy import deepcopy
from typing import Any, Mapping, Sequence
from collections.abc import Mapping, Sequence

from upath import UPath

from ..utils import BindPermission, format_bwrap_cmd
from .worker_config import NBAgentWorkerConfig


def get_project_root():
    """Get project root by running pixi info command."""
    pixi_info = subprocess.check_output(["pixi", "info"], text=True)
    for line in pixi_info.splitlines():
        if "Manifest file" in line:
            manifest_path = line.split()[2]
            return UPath(manifest_path).parent
    raise RuntimeError("Could not determine project root")


def _render(v: Any) -> str:
    if isinstance(v, Sequence) and not isinstance(v, str):
        return f"[{','.join(_render(x) for x in v if x is not None)}]"
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)

    if not s or any(c in " ,=[]{}:'\"" for c in s):
        return json.dumps(s, ensure_ascii=False)
    else:
        return s


def _flatten(cfg: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Flattens a dictionary into a list of Hydra override strings, each safely quoted for shell execution."""
    tokens: list[str] = []

    for key, value in cfg.items():
        if value is None:
            continue

        full_key = f"{prefix}{key}"

        if isinstance(value, Mapping):
            tokens.extend(_flatten(value, full_key + "."))
        else:
            if key == "task":
                rendered_value = _render(value).replace("${", r"\${")
            else:
                rendered_value = _render(value)
            override_str = f"{full_key}={shlex.quote(rendered_value)}"
            tokens.append(override_str)

    return tokens


def extract_shell_cmd_args(cfg: Mapping[str, Any]) -> str:
    """Convert *cfg* into a space-joined string of Hydra CLI overrides, each one already shell-escaped via `shlex.quote`."""
    return " ".join(_flatten(cfg)) + " "


def get_run_nb_agent_worker_cmd(config: NBAgentWorkerConfig):
    """Generate the bwrap command and shell command to run inside the sandbox.

    Args:
        config: Configuration for the NB Agent worker.

    Returns:
        A dictionary containing the bwrap command and metadata about the container paths.
    """
    config = deepcopy(config)

    project_root = get_project_root()
    external_lib_path = project_root / "external"
    src_path = project_root / "src"
    pixi_env_path = UPath(sys.executable).parent.parent

    container_workspace_path = UPath("/root/workspace")
    container_jupyter_path = container_workspace_path / "share" / "jupyter"
    container_agent_workspace_path = container_workspace_path / "agent"
    container_traj_save_storage_path = container_workspace_path / "traj_save_storage"
    container_log_path = container_workspace_path / "logs"

    file_binds: dict[UPath, tuple[UPath, BindPermission]] = {}
    if config.task_run.file_path:
        p = config.task_run.file_path
        if p:
            src = UPath(p).expanduser()
            if not src.is_absolute():
                src = (project_root / src).resolve()
            if not src.is_absolute():
                raise ValueError(f"file_path '{p}' could not be resolved to an absolute path")
            if not src.exists():
                raise FileNotFoundError(f"Provided file_path '{src}' does not exist")
            file_binds[src] = (UPath(src.as_posix()), BindPermission.R)

    # Build PYTHONPATH with all external libraries and project source
    pythonpath = []
    for d in external_lib_path.glob("*/"):
        if d.is_dir():
            container_path = container_workspace_path / "external" / d.name
            pythonpath.append(str(container_path))

    pythonpath.append(str(container_workspace_path / "src"))
    pythonpath_str = ":".join(pythonpath)

    # Create host log directory if it doesn't exist
    if config.log_path is not None and not config.log_path.exists():
        config.log_path.mkdir(parents=True, exist_ok=True)

    bind_paths = {
        "/usr": ("/usr", BindPermission.R),
        "/lib": ("/lib", BindPermission.R),
        "/lib64": ("/lib64", BindPermission.R),
        "/bin": ("/bin", BindPermission.R),
        "/sbin": ("/sbin", BindPermission.R),
        "/etc/alternatives": ("/etc/alternatives", BindPermission.R),
        "/etc/ssl": ("/etc/ssl", BindPermission.R),
        "/etc/ca-certificates": ("/etc/ca-certificates", BindPermission.R),
        "/etc/resolv.conf": ("/etc/resolv.conf", BindPermission.R),
        pixi_env_path: (container_workspace_path / "env", BindPermission.R),
        external_lib_path: (container_workspace_path / "external", BindPermission.R),
        src_path: (container_workspace_path / "src", BindPermission.R),
    }

    bind_paths.update(file_binds)

    if config.nb_agent.traj_save_storage_path and config.nb_agent.traj_save_storage_path.protocol != "s3":
        bind_paths[config.nb_agent.traj_save_storage_path] = (
            container_traj_save_storage_path,
            BindPermission.RW,
        )
        config.nb_agent.traj_save_storage_path = container_traj_save_storage_path
    if config.log_path is not None:
        if config.log_path.protocol == "s3":
            raise ValueError("Log path cannot be set to obs url.")
        bind_paths[config.log_path] = (container_log_path, BindPermission.RW)
        config.log_path = container_log_path

    new_dirs = [
        "tmp",
        "var",
        "root",
        container_workspace_path,
        container_jupyter_path,
        container_agent_workspace_path,
    ]

    for _, (dest, _) in file_binds.items():
        parent = dest.parent
        while parent != parent.root and parent not in new_dirs:
            new_dirs.append(parent)
            parent = parent.parent

    env_vars = {
        "HOME": "/root",
        "PATH": f"{container_workspace_path}/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": pythonpath_str,
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "JINA_API_KEY": os.environ.get("JINA_API_KEY", ""),
        "SERPER_API_KEY": os.environ.get("SERPER_API_KEY", ""),
        "VLM_MODEL_ID": os.environ.get("VLM_MODEL_ID", ""),
        "VLM_API_BASE": os.environ.get("VLM_API_BASE", ""),
        "VLM_API_KEY": os.environ.get("VLM_API_KEY", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", ""),
        "FIRECRAWL_BASE_URL": os.environ.get("FIRECRAWL_BASE_URL", ""),
        # Add Hydra specific environment variables
        "HYDRA_FULL_ERROR": "1",
        "HYDRA_LOG_PATH": str(container_log_path),
    }

    config.nb_agent.virtual_kernel_spec_path = container_jupyter_path
    config.nb_agent.agent_workspace = container_workspace_path

    # The shell command to run inside the sandbox
    shell_cmd = f"python -m l0.traj_sampler.nb_agent_sampler.worker "
    # Convert config to dict for iteration
    config_dict = config.to_dict()

    shell_cmd += extract_shell_cmd_args(config_dict)

    # Add Hydra output directory override
    shell_cmd += (
        f"hydra.run.dir={shlex.quote(f'{container_log_path}/{config.nb_agent.uuid}_${{now:%Y-%m-%d_%H-%M-%S}}')}"
    )

    bwrap_cmd = format_bwrap_cmd(bind_paths, new_dirs, env_vars, shell_cmd)

    return bwrap_cmd
