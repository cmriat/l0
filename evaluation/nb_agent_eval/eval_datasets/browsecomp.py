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
"""Natural Questions (NQ) dataset loader."""

import base64
import hashlib

import torch
from verl import DataProto
from upath import UPath
from datasets import load_dataset
from pydantic import Field, BaseModel, ValidationError

from ._base import BaseDataset
from .dataset_registry import DatasetRegistry


class BrowseCompSample(BaseModel):
    """Model for a BrowseComp Questions record."""

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., description="Raw question text")
    golden_answers: list[str] = Field(..., description="List of reference answers")
    problem_topic: str = Field(..., description="Problem topic")


@DatasetRegistry.register("browsecomp")
class BrowseCompDataset(BaseDataset):
    """Dataset loader for BrowseComp stored in JSONL format.

    Implements the BaseDataset interface:
    - `_load_all`: populates `self._splits` for train/dev/test.
    - `build_batch`: creates a DataProto batch + answer labels.
    """

    _FILENAMES: dict[str, str] = {"test": "browsecomp/test.jsonl"}

    _HF_PATH: dict[str, str] = {"test": "cmriat/browsecomp"}

    def __init__(self, data_dir: str | UPath, prompt: str | None = None, source: str = "hf") -> None:
        super().__init__(data_dir)
        self._prompt = prompt or "Answer the question in a word or a short phrase."
        self.eval_split = "test"

        if source == "hf":
            self._load_hf()
        elif source == "local":
            self._load_local()

    def _load_hf(self):
        for split, rel_path in self._HF_PATH.items():
            ds = load_dataset(rel_path)[self.eval_split]
            samples = []
            for sample in ds.to_list():
                samples.append(BrowseCompSample.model_validate(sample))

            self._splits[split] = samples

    def _load_local(self) -> None:
        """Load and validate all JSONL splits into `self._splits`."""
        for split, rel_path in self._FILENAMES.items():
            path = UPath(self._root) / rel_path
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {path}")

            samples: list[BrowseCompSample] = []
            with path.open("r", encoding="utf-8") as fp:
                for lineno, raw in enumerate(fp, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        samples.append(BrowseCompSample.model_validate_json(raw))
                    except ValidationError as err:
                        raise ValueError(
                            f"Invalid JSONL entry in {path}@{lineno}: {err}"  # type: ignore
                        ) from err
            self._splits[split] = samples

    def build_batch(
        self,
    ) -> tuple[DataProto, list[list[str]]]:
        """Convert a split into a DataProto batch and corresponding labels."""
        dataset = self.get_split(self.eval_split)

        uuids: list[str] = []
        prompts: list[str] = []
        labels: list[list[str]] = []
        for entry in dataset:
            format_id = str(entry.id).replace("_", "-")
            uuids.append(format_id)
            prompts.append(f"{entry.question}. {self._prompt}")
            labels.append(entry.golden_answers)

        non_tensors = {"uuids": uuids, "question": prompts, "file_paths": [None for _ in range(len(uuids))]}
        batch = DataProto.from_dict(
            tensors={"dummy": torch.zeros(len(dataset), 1)},  # placeholder tensor
            non_tensors=non_tensors,
        )
        return batch, labels


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key, strict=False))
    return decrypted.decode()
