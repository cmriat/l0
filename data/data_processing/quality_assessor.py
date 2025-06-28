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

"""Assessor for evaluating datasets."""

import os
import re
import json
import atexit  # Add this import
import asyncio
import concurrent

import lance
import aiohttp  # Added import
import backoff
import pyarrow as pa
from tqdm import tqdm
from lance import LanceDataset
from upath import UPath

from l0.utils import get_num_processes

from .utils import (
    _log_giveup,
    _log_backoff,
    validate_lance_dataset,
)
from .prompt import QUALITY_ASSESSOR_PROMPT


@backoff.on_exception(backoff.expo, Exception, max_tries=3, on_backoff=_log_backoff, on_giveup=_log_giveup)
async def make_openai_request_async(qa_pair: str, api_url: str, api_key: str) -> str:
    """Makes an async request to the OpenAI API with backoff using aiohttp."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": os.getenv("OPENAI_MODEL_ID"),
        "messages": [
            {"role": "system", "content": QUALITY_ASSESSOR_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": qa_pair}]},
        ],
    }
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors
            completion = await response.json()
            return completion["choices"][0]["message"]["content"]


class DataAssessor:
    ASSESSMENT_COLUMN_DEFAULTS = {
        "assessed": False,
        "objectivity rating": -2,
        "objectivity rationale": [""],
        "temporal stability rating": -2,
        "temporal stability rationale": [""],
    }

    def __init__(
        self,
        merged_dataset: LanceDataset | UPath,
        assessed_dataset_dir: UPath,
        save_interval: int = 1000,
        num_proc: int = 8,
        batch_size: int = 10,
    ):
        # Handle both LanceDataset objects and paths to Lance datasets
        if isinstance(merged_dataset, LanceDataset):
            self.merged_dataset = merged_dataset
        else:
            # Load from disk if a path is provided
            if not validate_lance_dataset(merged_dataset):
                raise ValueError(f"Invalid LanceDataset at {merged_dataset}")
            self.merged_dataset = lance.dataset(str(merged_dataset))

        self.num_proc = get_num_processes(num_proc)
        self.assessed_data_dir = assessed_dataset_dir
        self.assessed_dataset = None
        self.save_interval = save_interval
        self.batch_size = batch_size
        if self.batch_size > self.save_interval:
            raise ValueError("Batch size cannot be larger than save interval.")
        # Store API URL and key for aiohttp requests
        self.api_url = os.getenv("OPENAI_API_BASE")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_proc)

        atexit.register(self.shutdown)

    def get_assessed_dataset(self) -> None:
        assessed_data_dir_str = str(self.assessed_data_dir)

        if not validate_lance_dataset(self.assessed_data_dir):
            # Copy the input LanceDataset to the assessed dataset directory
            self.assessed_data_dir.mkdir(parents=True, exist_ok=True)

            # Convert the merged dataset to a PyArrow table and write as new LanceDataset
            table = self.merged_dataset.to_table()
            assessed_lance_dataset = lance.write_dataset(table, assessed_data_dir_str, mode="overwrite")
            print(f"Created assessed LanceDataset at {assessed_data_dir_str}")
        else:
            assessed_lance_dataset = lance.dataset(assessed_data_dir_str)

        self.assessed_dataset = assessed_lance_dataset

    @staticmethod
    def parse_assessment(assessment_text):
        """Parse the assessment text into a structured dictionary with four specific components."""
        # Try to extract and parse JSON format first
        json_str = None

        # First try to find JSON content between ```json and ``` tags
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        json_match = re.search(json_pattern, assessment_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no match, try alternative pattern without code blocks
            json_pattern = r"\{[\s\S]*?\}"
            json_match = re.search(json_pattern, assessment_text)
            if json_match:
                json_str = json_match.group(0)

        data = json.loads(json_str)

        return data

    @staticmethod
    @backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=_log_backoff, on_giveup=_log_giveup)
    async def _assess_sample_internal(
        sample: dict,
        loop: asyncio.AbstractEventLoop,
        process_pool: concurrent.futures.ProcessPoolExecutor,
        api_url: str,
        api_key: str,
    ) -> dict:
        """Internal method to assess a sample with backoff decorator using aiohttp."""
        qa_pair = f"Question: {sample['question']}\\nAnswer: {sample['answer']}"

        assessment_text = await make_openai_request_async(qa_pair, api_url, api_key)
        assessment_dict = await loop.run_in_executor(process_pool, DataAssessor.parse_assessment, assessment_text)

        # Update the sample with assessment results
        sample["objectivity rating"] = assessment_dict["objectivity"]["rating"]
        sample["objectivity rationale"] = assessment_dict["objectivity"]["rationale"]
        sample["temporal stability rating"] = assessment_dict["temporal_stability"]["rating"]
        sample["temporal stability rationale"] = assessment_dict["temporal_stability"]["rationale"]
        sample["assessed"] = True

        return sample

    @staticmethod
    async def assess_sample_async(
        sample: dict,
        loop: asyncio.AbstractEventLoop,
        process_pool: concurrent.futures.ProcessPoolExecutor,
        api_url: str,
        api_key: str,
    ) -> dict:
        """Assess a single sample asynchronously with graceful failure handling using aiohttp."""
        if sample.get("assessed", False):
            return sample

        try:
            return await DataAssessor._assess_sample_internal(sample, loop, process_pool, api_url, api_key)
        except Exception as e:
            # If all retries fail, return the sample with default values
            print(f"Failed to assess sample after all retries: {str(e)}")
            for col_name, default_value in DataAssessor.ASSESSMENT_COLUMN_DEFAULTS.items():
                sample[col_name] = default_value
            return sample

    def process_batch(self, table_batch: list[dict]):
        """Process a batch of samples (as a PyArrow Table) synchronously, with internal async handling."""

        async def _process_batch_async_with_session():
            loop = asyncio.get_running_loop()
            tasks = []
            for sample_dict in table_batch:
                if not sample_dict.get("assessed", False):
                    tasks.append(
                        self.assess_sample_async(sample_dict, loop, self.process_pool, self.api_url, self.api_key)
                    )
                else:
                    # For already assessed samples, create a completed future
                    future = asyncio.Future()
                    future.set_result(sample_dict)
                    tasks.append(future)
            return await asyncio.gather(*tasks)

        # Run the async processing within the existing event loop if possible, or create a new one
        try:
            results_list_of_dicts = asyncio.run(_process_batch_async_with_session())
        except RuntimeError:  # If there's already a running loop
            results_list_of_dicts = asyncio.new_event_loop().run_until_complete(_process_batch_async_with_session())

        return results_list_of_dicts

    def assess(self) -> bool:
        """Assess all datasets, saving progress periodically."""
        print(f"Using {self.num_proc} processes for inference.")
        # Lazy loading of the assessed dataset
        if self.assessed_dataset is None:
            self.get_assessed_dataset()
            if self.assessed_dataset is None:
                print("Failed to load or create dataset shards.")
                return None
        pbar = tqdm(total=len(self.assessed_dataset), desc="Processing batches", unit="batch")

        for raw_batch in self.assessed_dataset.to_batches(batch_size=self.save_interval):
            # Process the batch
            batch_data = raw_batch.to_pylist()
            start_idx = 0
            for end_idx in range(self.batch_size, len(batch_data), self.batch_size):
                batch_data[start_idx:end_idx] = self.process_batch(batch_data[start_idx:end_idx])
                pbar.update(end_idx - start_idx)
                start_idx = end_idx

            # Convert processed batch back to PyArrow Table
            processed_batch_table = pa.Table.from_pylist(batch_data, schema=raw_batch.schema)
            # update and insert
            (
                self.assessed_dataset.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(processed_batch_table)
            )

        pbar.close()
        return True

    def shutdown(self):
        """Shutdown the process pool executor."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
