# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# This file is adapted from the sglang library.
# Copyright 2023-2024 SGLang Team
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
"""This module provides an adapter for the SGLang engine, supporting both direct engine and HTTP server modes.

It includes classes for managing model generation, weight updates, and memory control in a distributed setting.
"""

# Standard library imports
import os
import time
import logging
import multiprocessing
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Tuple,
    Union,
    Literal,
    Iterable,
    Iterator,
    Optional,
)

import torch

# Third-party imports
import requests
import torch.distributed as dist
from PIL.Image import Image
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    kill_process_tree,
)
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.server_args import ServerArgs
from torch.distributed.tensor import DTensor, DeviceMesh
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.model_executor.model_runner import LocalSerializedTensor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class VerlEngine:
    def __init__(
        self,
        device_mesh_cpu: DeviceMesh,
        nnodes: int = 1,
        backend: Literal["engine", "server"] = "engine",
        **kwargs,
    ):
        monkey_patch_torch_reductions()
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu.get_local_rank()
        self._rank = device_mesh_cpu.get_rank()
        self._tp_size = device_mesh_cpu.size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        # Common engine keyword arguments
        engine_kwargs = dict(**kwargs, tp_size=self._tp_size, node_rank=node_rank, nnodes=nnodes)

        if backend == "engine":
            if first_rank_in_node:
                os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
                self._engine = Engine(**engine_kwargs)
            else:
                self._engine = None

        elif backend == "server":
            if self._tp_rank == 0:
                self._engine = HttpServerEngineAdapter(**engine_kwargs)
            else:
                self._engine = None
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        dist.barrier(group=self._device_mesh_cpu.get_group())

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.

        Please refer to `GenerateReqInput` for the documentation.
        """
        if self._tp_rank == 0:
            output = self._engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
            )
        else:
            output = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu.get_group(),
            src=self._device_mesh_cpu.mesh[0].item(),
            force_cpu_device=False,
        )

        return output

    def update_weights_from_tensor(
        self,
        named_tensors: Iterable[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
    ):
        # Most naive implementation, can optimize a lot if it is bottleneck
        for _, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self._tp_rank == 0:
                gathered_serialized_tensors = [None for _ in range(self._tp_size)]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self._device_mesh_cpu.mesh.tolist()[0],
                group=self._device_mesh_cpu.get_group(),
            )

            if self._tp_rank == 0:
                self._engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

        if self._tp_rank == 0:
            self._engine.flush_cache()

    def release_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.abort_all_requests()
            self._engine.release_memory_occupation()

    def resume_memory_occupation(self):
        if self._tp_rank == 0:
            self._engine.resume_memory_occupation()

    def shutdown(self):
        if self._engine is not None:
            self._engine.shutdown()

    def abort_all_requests(self):
        if self._tp_rank == 0 and isinstance(self._engine, HttpServerEngineAdapter):
            self._engine.abort_all_requests()


class EngineBase(ABC):
    """Abstract base class for engine interfaces that support generation, weight updating, and memory control.

    This base class provides a unified API for both HTTP-based engines and engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """Generate outputs based on given inputs."""
        pass

    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights with in-memory tensor data."""
        pass

    @abstractmethod
    def release_memory_occupation(self):
        """Release GPU memory occupation temporarily."""
        pass

    @abstractmethod
    def resume_memory_occupation(self):
        """Resume GPU memory occupation which is previously released."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass


def _launch_server_with_patches(server_args: ServerArgs):
    """Launch server with monkey patches applied in the subprocess."""
    # Apply monkey patches in the subprocess
    from . import monkey_patch_sglang

    monkey_patch_sglang()
    launch_server(server_args)


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    p = multiprocessing.Process(target=_launch_server_with_patches, args=(server_args,))
    p.start()

    base_url = server_args.url()
    timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
    start_time = time.time()

    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {server_args.api_key}",
                }
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    return p
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

    p.terminate()
    raise TimeoutError("Server failed to start within the timeout period.")


class HttpServerEngineAdapter(EngineBase):
    """You can use this class to launch a server from a VerlEngine instance.

    We recommend using this class only you need to use http server.
    Otherwise, you can use Engine directly.
    """

    def __init__(self, **kwargs):
        self.server_args = ServerArgs(**kwargs)
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}")
        self.process = launch_server_process(self.server_args)

    def _make_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return response.text

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                    for _ in range(self.server_args.tp_size)
                ],
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def shutdown(self):
        kill_process_tree(self.process.pid)

    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload)

    def release_memory_occupation(self):
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self):
        return self._make_request("resume_memory_occupation")

    def flush_cache(self):
        return self.abort_all_requests()

    def abort_all_requests(self):
        self._make_request("abort_request", {"rid": ""})
        # check if aborted with flush cache
        retry_count = 0
        while True:
            try:
                logger.warning("Begin flushing cache")
                return self._make_request("flush_cache")
            except Exception as err:
                logger.warning(f"Flushing cache failed, retrying... {retry_count}")
                time.sleep(0.1)
                retry_count += 1
                if retry_count > 10:
                    raise Exception("Flushing cache failed") from None
