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
"""Test the tokenize endpoint of the HTTP server engine."""

import time
import asyncio
import multiprocessing

import requests
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from l0.sglang_adapter import HttpServerEngineAdapter

LOCAL_DIR = snapshot_download("Qwen/Qwen3-0.6B")

multiprocessing.set_start_method("spawn", force=True)


# TODO: use only one engine instance for all tests
def test_monkey_patch_tokenize_endpoint():
    engine = HttpServerEngineAdapter(model_path=LOCAL_DIR)

    url = f"http://localhost:30000/tokenize"

    text = "Hello world"
    target_token_ids = [9707, 1879]

    response = requests.post(url, json={"text": text})

    engine.shutdown()

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    response = response.json()

    assert response["token_ids"][0] == target_token_ids[0], (
        f"Expected token ID {target_token_ids[0]}, got {response['token_ids'][0]}"
    )
    assert response["token_ids"][1] == target_token_ids[1], (
        f"Expected token ID {target_token_ids[1]}, got {response['token_ids'][1]}"
    )
    assert response["text"] == text, f"Expected text '{text}', got '{response['text']}'"


def test_monkey_patch_detokeniz_endpoint():
    engine = HttpServerEngineAdapter(model_path=LOCAL_DIR)

    url = f"http://localhost:30000/detokenize"

    target_text = "Hello world"

    response = requests.post(url, json={"token_ids": [9707, 1879]})

    engine.shutdown()

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    response = response.json()

    assert response["text"] == target_text, f"Expected text '{target_text}', got '{response['text']}'"


def test_monkey_patch_tokenize_with_format_endpoint():
    engine = HttpServerEngineAdapter(model_path=LOCAL_DIR)

    msg = [
        {"role": "system", "content": "You are a helpful assistant that responds with concise answers."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)

    target_token_ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=False)
    target_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

    url = f"http://localhost:30000/tokenize_with_template"

    response = requests.post(url, json={"messages": msg, "model": "Qwen/Qwen2.5-1.5B-Instruct"})

    engine.shutdown()

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    response = response.json()
    assert response["token_ids"] == target_token_ids, (
        f"Expected token IDs {target_token_ids}, got {response['token_ids']}"
    )
    assert response["text"] == target_text, f"Expected text '{target_text}', got '{response['text']}'"


async def send_single_request():
    try:
        # Using a stop sequence that will likely never be met to make the request long-running
        # Also, ensure the prompt asks for a lengthy response.
        client = AsyncOpenAI(base_url=f"http://localhost:30000/v1", api_key="None")
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",  # Matches LOCAL_DIR content
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Tell me a very, very long story that will take a significant amount of time and tokens to generate.",
                },
            ],
            stop="THIS_IS_AN_UNLIKELY_SEQUENCE_TO_EVER_BE_GENERATED_BY_THE_MODEL_XYZABC123",
            stream=False,  # Non-streaming for simpler abort check
            timeout=30,  # Client-side timeout for the request
        )
        return response
    except Exception as e:
        return e


async def send_requests_concurrently():
    tasks = [send_single_request() for _ in range(20)]
    return await asyncio.gather(*tasks)


def request_submission():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    results = loop.run_until_complete(send_requests_concurrently())
    return results


def test_monkey_patch_abort_all_requests():
    engine = HttpServerEngineAdapter(model_path=LOCAL_DIR, port=30000)

    req_p = multiprocessing.Process(target=request_submission)
    req_p.start()

    # Allow some time for requests to be initiated and become in-flight
    time.sleep(1)

    print("Aborting all requests...")
    engine.abort_all_requests()

    try:
        print(f"Flush cache response: {engine._make_request('flush_cache', {})}")
    except Exception as e:
        print(f"Error flushing cache: {e}")

    req_p.terminate()
    engine.shutdown()


if __name__ == "__main__":
    test_monkey_patch_tokenize_endpoint()
    test_monkey_patch_detokeniz_endpoint()
    test_monkey_patch_tokenize_with_format_endpoint()
    test_monkey_patch_abort_all_requests()
