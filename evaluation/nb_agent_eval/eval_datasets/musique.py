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
"""Musique dataset loader."""

from typing import Any

import torch
from verl import DataProto
from upath import UPath
from datasets import load_dataset
from pydantic import Field, BaseModel, ValidationError

from ._base import BaseDataset
from .dataset_registry import DatasetRegistry


class MusiqueSample(BaseModel):
    """Model for a single Musique question record."""

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., description="Raw question text")
    golden_answers: list[str] = Field(..., description="List of reference answers")
    metadata: dict[str, Any] = Field(..., description="Metadata")


@DatasetRegistry.register("musique")
class MusiqueDataset(BaseDataset):
    """Dataset loader for Musique stored in JSONL format.

    Implements the BaseDataset interface:
    - `_load_all`: populates `self._splits` for train/dev/test.
    - `build_batch`: creates a DataProto batch + answer labels.
    """

    _FILENAMES: dict[str, str] = {
        "validation": "musique/validation.jsonl",
    }

    _HF_PATH: dict[str, str] = {"validation": "cmriat/musique"}

    def __init__(self, data_dir: str | UPath, prompt: str | None = None, source: str = "hf") -> None:
        super().__init__(data_dir)
        self._prompt = (
            prompt
            or "The question is a multi-hop question, you should it breakdown first and then solve it. Answer the question in a word or a short phrase."
        )
        self.eval_split = "validation"

        if source == "hf":
            self._load_hf()
        elif source == "local":
            self._load_local()

    def _load_hf(self):
        for split, rel_path in self._HF_PATH.items():
            ds = load_dataset(rel_path)[self.eval_split]
            samples = []
            for sample in ds.to_list():
                samples.append(MusiqueSample.model_validate(sample))

            self._splits[split] = samples

    def _load_local(self) -> None:
        """Load and validate all JSONL splits into `self._splits`."""
        for split, rel_path in self._FILENAMES.items():
            path = UPath(self._root) / rel_path
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            samples: list[MusiqueSample] = []
            with path.open("r", encoding="utf-8") as fp:
                for lineno, raw in enumerate(fp, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        samples.append(MusiqueSample.model_validate_json(raw))
                    except ValidationError as err:
                        raise ValueError(
                            f"Invalid JSONL entry in {path}@{lineno}: {err}"  # type: ignore
                        ) from err
            self._splits[split] = samples

    def build_batch(self) -> tuple[DataProto, list[list[str]]]:
        """Convert a split into a DataProto batch and corresponding labels."""
        dataset = self.get_split(self.eval_split)

        uuids: list[str] = []
        prompts: list[str] = []
        labels: list[list[str]] = []
        for entry in dataset:
            format_id = str(entry.id).replace("_", "-")
            uuids.append(format_id)
            prompts.append(f"{entry.question} {self._prompt}")
            labels.append(entry.golden_answers)

        non_tensors = {"uuids": uuids, "question": prompts, "file_paths": [None for _ in range(len(uuids))]}
        batch = DataProto.from_dict(
            tensors={"dummy": torch.zeros(len(dataset), 1)},  # placeholder tensor
            non_tensors=non_tensors,
        )
        return batch, labels
