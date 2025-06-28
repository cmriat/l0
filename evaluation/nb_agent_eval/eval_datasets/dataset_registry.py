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

"""Dataset registry for managing and creating dataset instances."""

from typing import Type, Callable

from ._base import BaseDataset


class DatasetRegistry:
    _registry: dict[str, BaseDataset] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseDataset]], Type[BaseDataset]]:
        def _decorator(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
            if name in cls._registry:
                raise KeyError(f"Dataset '{name}' already registered.")
            if not issubclass(dataset_cls, BaseDataset):
                raise TypeError("Registered class must inherit from BaseDataset.")
            cls._registry[name] = dataset_cls
            return dataset_cls

        return _decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        try:
            return cls._registry[name]
        except KeyError:
            valid = ", ".join(sorted(cls._registry))
            raise KeyError(f"Dataset '{name}' not found. Valid options: [{valid}]")  # noqa: B904

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> BaseDataset:
        """Convenience factory: DatasetRegistry.create("nq", data_dir)"""  # noqa: D415
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._registry)
