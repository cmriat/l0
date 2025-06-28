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

"""Monkey patch the DataProto.concat function to filter out empty DataProto items."""

from typing import List

import numpy as np
import torch
from tensordict import TensorDictBase
from verl.protocol import DataProto, list_of_dict_to_dict_of_list


def monkey_patch_proto_concat():
    def concat(data: List["DataProto"]) -> "DataProto":
        # Filter out empty DataProto items
        data = [item for item in data if (item.batch is not None or item.meta_info is not None)]
        if not data:
            return DataProto()
        # original concat function
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    DataProto.concat = concat


def monkey_patch_tensordict_chunk():
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]:
        if chunks < 1:
            raise ValueError(f"chunks must be a strictly positive integer, got {chunks}.")
        total_size = self.batch_size[dim]
        base_size = total_size // chunks
        extra = total_size % chunks

        result = []
        start = 0
        for i in range(chunks):
            if i < extra:
                chunk_size = base_size + 1
            else:
                chunk_size = base_size
            end = start + chunk_size
            if dim == 0:
                # 对于第0维，直接使用切片
                result.append(self[start:end])
            else:
                # 对于其他维度，需要构造索引
                index = [slice(None)] * len(self.batch_size)
                index[dim] = slice(start, end)
                result.append(self[tuple(index)])
            start = end
        return tuple(result)

    TensorDictBase.chunk = chunk
