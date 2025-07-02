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
"""Evaluation orchestrator for NB agent experiments."""

import os
import json
import asyncio
import logging
from collections import defaultdict

import aiohttp
from upath import UPath
from sampler import NBAgentEvalSampler
from config_dataclass import ConfigDataclass
from eval_datasets._base import BaseDataset
from eval_datasets.dataset_registry import DatasetRegistry

VALID_DATASETS = ["bamboogle", "musique", "hotpotqa", "simpleqa"]

MAX_CONCURRENCY = 32

logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)


class NBAgentEvaluator:
    """Runs batched inference and writes results, including UUID and question."""

    def __init__(self, config: ConfigDataclass, dataset_names: list[str] | None = None) -> None:
        self.save_dir = UPath(config.evaluator.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_names = dataset_names or DatasetRegistry.list_available()
        self.n_samples = config.evaluator.n_samples

        self.datasets: dict[str, BaseDataset] = {
            name: DatasetRegistry.create(
                name, UPath(config.evaluator.data_dir), source="hf" if config.evaluator.data_dir == "hf" else "local"
            )
            for name in dataset_names
        }
        self.sampler = NBAgentEvalSampler(config=config)
        logger.info(config)

    def run(self, dataset_names: list[str]) -> dict[str, list[dict[str, str]]]:
        """Evaluate *dataset_names* and return mapping ``name -> records``.

        Each record includes: uuid, question, model answer, and label.
        """
        all_results: dict[str, list[dict[str, str]]] = {}
        for name in dataset_names:
            if name not in self.datasets:
                logger.warning("Unknown dataset %s - skipped", name)
                continue
            eval_result_dir = os.path.join(self.save_dir, name)
            try:
                logger.info("Evaluating %s", name)
                eval_dataset = self.datasets[name]

                eval_dataset._load_hf()

                batch, labels = eval_dataset.build_batch()
                predictions_sampled = defaultdict(list)
                for _ in range(self.n_samples):
                    predictions = self.sampler.batch_inference(batch, eval_result_dir=eval_result_dir)
                    for uuid, pred in predictions.items():
                        predictions_sampled[uuid].append(str(pred))

                non_tensors = batch.non_tensor_batch
                uuids: list[str] = non_tensors.get("uuids", [])
                questions: list[str] = non_tensors.get("question", [])

                records: list[dict[str, str]] = []

                for u, q, g in zip(uuids, questions, labels, strict=True):
                    records.append(
                        {
                            "uuid": u,
                            "question": q,
                            "answer": predictions_sampled[u],
                            "label": g,
                        }
                    )

                self._write_jsonl(eval_result_dir, records)
                all_results[name] = records
            except Exception as e:
                logging.error(f"Evaluate '{name}' failed, error: {e}")
        return all_results

    def _write_jsonl(self, eval_result_dir: str, records: list[dict[str, str]]) -> None:
        """Write records to ``<save_dir>/<dataset_name>.jsonl`` using UPath."""
        os.makedirs(os.path.join(eval_result_dir), exist_ok=True)
        save_path = os.path.join(eval_result_dir, "result.jsonl")
        with open(save_path, "w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Saved %d records to %s", len(records), save_path)


class DirectEvaluator:
    """Runs batched, asynchronous inference via HTTP API and writes results."""

    def __init__(self, config, dataset_names) -> None:
        self.save_dir = UPath(config.evaluator.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = config.evaluator.n_samples

        dataset_names = dataset_names or DatasetRegistry.list_available()

        self.datasets: dict[str, BaseDataset] = {
            name: DatasetRegistry.create(name, UPath(config.evaluator.data_dir)) for name in dataset_names
        }

        self.api_base = config.agent.openai_server_model.base_url or os.getenv("OPENAI_API_BASE")
        if self.api_base is None:
            raise ValueError("`api_base` is not provided and OPENAI_API_BASE is not set.")

        self.model_name = config.agent.openai_server_model.model_id
        self.concurrency = min(config.max_concurrency, MAX_CONCURRENCY)
        self.timeout = config.task_timeout
        self.n_samples = config.evaluator.n_samples

        self.api_url = os.path.join(self.api_base, "chat/completions")
        self.headers = {"Content-Type": "application/json"}

    def run(self, dataset_names: list[str]) -> dict[str, list[dict[str, str | list[str]]]]:
        """Evaluate the given *dataset_names*.

        Returns:
        -------
        dict
            Mapping ``dataset_name -> records`` where each *records* is the list
            of JSON-serialisable dicts saved to ``<save_dir>/<dataset>/result.jsonl``.
        """
        all_results: dict[str, list[dict[str, str | list[str]]]] = {}

        for name in dataset_names:
            if name not in self.datasets:
                logger.warning("Unknown dataset %s - skipped", name)
                continue

            eval_result_dir = os.path.join(self.save_dir, name)

            try:
                logger.info("Evaluating %s", name)
                batch, labels = self.datasets[name].build_batch()

                non_tensors = batch.non_tensor_batch
                uuids: list[str] = non_tensors.get("uuids", [])
                questions: list[str] = non_tensors.get("question", [])

                samples: list[dict[str, str | list[str]]] = [
                    {
                        "id": u,
                        "question": q,
                        "golden_answers": g,
                    }
                    for u, q, g in zip(uuids, questions, labels, strict=True)
                ]

                predictions_sampled: dict[str, list[str]] = defaultdict(list)
                for _ in range(self.n_samples):
                    results = asyncio.run(
                        self._run_async_eval(samples),
                    )
                    for r in results:
                        predictions_sampled[r["uuid"]].append(r["pred"])

                records: list[dict[str, str | list[str]]] = [
                    {
                        "uuid": u,
                        "question": q,
                        "answer": predictions_sampled[u],
                        "label": g,
                    }
                    for u, q, g in zip(uuids, questions, labels, strict=True)
                ]

                self._write_jsonl(eval_result_dir, records)
                all_results[name] = records

            except Exception as exc:
                logger.error("Evaluate '%s' failed: %s", name, exc)

        return all_results

    async def _query(self, session: aiohttp.ClientSession, item: dict) -> dict:
        """Single HTTP POST for one example, returns minimal prediction dict."""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": (item["question"] + ". Answer the question in a word or a short phrase."),
                }
            ],
        }
        async with session.post(
            self.api_url,
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            answer: str = data["choices"][0]["message"]["content"]
            return {
                "uuid": item["id"],
                "question": item["question"],
                "pred": answer.strip(),
                "label": item["golden_answers"],
            }

    async def _run_async_eval(
        self,
        dataset: list[dict],
    ) -> list[dict]:
        """Run asynchronous evaluation over *dataset* with set concurrency."""
        semaphore = asyncio.Semaphore(self.concurrency)

        async with aiohttp.ClientSession(headers=self.headers) as session:

            async def bound(item):
                async with semaphore:
                    return await self._query(session, item)

            tasks = [bound(d) for d in dataset]
            return await asyncio.gather(*tasks)

    def _write_jsonl(self, eval_result_dir: str | os.PathLike, records: list[dict]) -> None:
        """Write records to ``<save_dir>/<dataset_name>.jsonl`` using UPath."""
        os.makedirs(os.path.join(eval_result_dir), exist_ok=True)
        save_path = os.path.join(eval_result_dir, "result.jsonl")
        with open(save_path, "w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Saved %d records to %s", len(records), save_path)
