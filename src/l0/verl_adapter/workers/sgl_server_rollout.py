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
"""Adapt SGLang server rollout for Agentic RL."""

import os

import torch.distributed
from verl import DataProto
from torch import nn
from omegaconf import DictConfig
from sglang.srt.utils import get_ip, broadcast_pyobj
from sglang.srt.server_args import PortArgs, ServerArgs
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.workers.rollout.base import BaseRollout
from torch.distributed.device_mesh import init_device_mesh

from l0.sglang_adapter import VerlEngine


class SGLServerRollout(BaseRollout):
    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        tp_size = tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )

        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        # device_mesh_device = init_device_mesh("cuda", **device_mesh_kwargs)

        # get tp_rank of this process in this tp group
        tp_rank = device_mesh_cpu["tp"].get_local_rank()
        visible_devices = [None] * device_mesh_cpu.size(1)
        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], device_mesh_cpu.get_group("tp")
        )
        visible_devices_set = set(visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(visible_devices_set)))

        nnodes = -(-tp_size // len(visible_devices_set))
        base_port = self.config.get("base_port", 8000)
        self.port = base_port + device_mesh_cpu["dp"].get_local_rank()
        server_args = ServerArgs(model_path=actor_module, nnodes=nnodes, port=self.port)
        if nnodes > 1:
            ip, port_args = get_ip(), PortArgs.init_new(server_args)
            [ip, port_args] = broadcast_pyobj(
                [ip, port_args],
                rank=tp_rank,
                dist_group=device_mesh_cpu.get_group("tp"),
                src=device_mesh_cpu["tp"].mesh[0].item(),
            )
            dist_init_addr = f"{ip}:{port_args.nccl_port}"
        else:
            dist_init_addr = None
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        self.inference_engine = VerlEngine(
            model_path=actor_module,
            dtype=config.dtype,
            mem_fraction_static=config.gpu_memory_utilization,
            device_mesh_cpu=device_mesh_cpu["tp"],
            enable_memory_saver=True,
            base_gpu_id=0,
            gpu_id_step=1,
            load_format=load_format,
            dist_init_addr=dist_init_addr,
            nnodes=nnodes,
            backend="server",
            port=self.port,
            host="0.0.0.0",
            enable_mixed_chunk=True,
            log_level="warning",
            # log_requests=True,
            # log_requests_level=2,
            # max_running_requests=1,
        )

        # offload
        self.inference_engine.release_memory_occupation()

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        raise NotImplementedError("Server backend should use request-response mode.")
