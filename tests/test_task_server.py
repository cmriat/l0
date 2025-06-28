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
"""Test the task server for managing asynchronous tasks."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from l0.traj_sampler.task_server import app, cleanup as app_cleanup


@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks():
    await app_cleanup()
    yield
    await app_cleanup()


@pytest.mark.asyncio
async def test_create_task_success(client: TestClient):
    task_cmd = ["echo", "hello"]
    task_uuid = "test_task_1"
    response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    await asyncio.sleep(0.5)

    status_response = client.get(f"/check_task_status/{task_uuid}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] in ["running", "completed"]
    if status_data["status"] == "completed":
        assert status_data["output"] == "hello\n"
        assert status_data["error"] == ""


@pytest.mark.asyncio
async def test_create_task_duplicate_uuid(client: TestClient):
    task_cmd = ["sleep", "0.2"]
    task_uuid = "test_task_duplicate"
    client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})

    response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert response.status_code == 500

    await asyncio.sleep(0.5)

    response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_check_task_status_not_found(client: TestClient):
    response = client.get("/check_task_status/non_existent_task")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


@pytest.mark.asyncio
async def test_check_task_status_running_and_completed(client: TestClient):
    task_cmd = ["sleep", "0.2"]  # A task that runs for a short while
    task_uuid = "test_task_sleep"
    create_response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert create_response.status_code == 200

    status_response_running = client.get(f"/check_task_status/{task_uuid}")
    assert status_response_running.status_code == 200
    assert status_response_running.json()["status"] == "running"

    await asyncio.sleep(0.5)  # Wait for the task to complete

    status_response_completed = client.get(f"/check_task_status/{task_uuid}")
    assert status_response_completed.status_code == 200
    assert status_response_completed.json()["status"] == "completed"
    assert status_response_completed.json()["output"] == ""  # sleep command has no stdout
    assert status_response_completed.json()["error"] == ""


@pytest.mark.asyncio
async def test_task_failure(client: TestClient):
    task_cmd = ["false"]  # This command will fail
    task_uuid = "test_task_fail"
    response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert response.status_code == 200

    await asyncio.sleep(0.1)  # Allow time for failure

    status_response = client.get(f"/check_task_status/{task_uuid}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "failed"
    assert status_data["output"] == ""
    # stderr for `false` is typically empty, but the status is the key here


@pytest.mark.asyncio
async def test_close_task_success(client: TestClient):
    task_cmd = ["sleep", "1"]  # A long running task
    task_uuid = "test_task_to_close"
    client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})

    await asyncio.sleep(0.1)  # Ensure it's running

    close_response = client.post(f"/close_task/{task_uuid}")
    assert close_response.status_code == 200
    assert close_response.json() == {"status": "OK"}

    # Check if task is removed
    status_response = client.get(f"/check_task_status/{task_uuid}")
    assert status_response.status_code == 404  # Should be not found after closing


@pytest.mark.asyncio
async def test_close_task_not_found(client: TestClient):
    response = client.post("/close_task/non_existent_task_to_close")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"


@pytest.mark.asyncio
async def test_close_task_already_completed(client: TestClient):
    task_cmd = ["echo", "done"]
    task_uuid = "test_task_already_done"
    client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})

    await asyncio.sleep(0.2)  # Let it complete

    status_response = client.get(f"/check_task_status/{task_uuid}")
    assert status_response.json()["status"] == "completed"

    close_response = client.post(f"/close_task/{task_uuid}")
    assert close_response.status_code == 200


@pytest.mark.asyncio
async def test_list_tasks(client: TestClient):
    response_empty = client.get("/list_tasks")
    assert response_empty.status_code == 200
    assert response_empty.json() == {"tasks": []}

    client.post("/create_task", json={"cmd": ["echo", "1"], "uuid": "task_a"})
    client.post("/create_task", json={"cmd": ["echo", "2"], "uuid": "task_b"})

    await asyncio.sleep(0.1)  # allow tasks to register

    response_filled = client.get("/list_tasks")
    assert response_filled.status_code == 200
    tasks_list = response_filled.json()["tasks"]
    assert len(tasks_list) == 2
    assert "task_a" in tasks_list
    assert "task_b" in tasks_list


@pytest.mark.asyncio
async def test_cleanup_endpoint(client: TestClient):
    client.post("/create_task", json={"cmd": ["sleep", "1"], "uuid": "task_cleanup_1"})
    client.post("/create_task", json={"cmd": ["sleep", "1"], "uuid": "task_cleanup_2"})

    await asyncio.sleep(0.1)  # allow tasks to start

    list_resp_before = client.get("/list_tasks")
    assert len(list_resp_before.json()["tasks"]) == 2

    cleanup_response = client.post("/cleanup")
    assert cleanup_response.status_code == 200
    assert cleanup_response.json()["status"] == "OK"
    assert "All tasks cleaned up" in cleanup_response.json()["message"]

    list_resp_after = client.get("/list_tasks")
    assert len(list_resp_after.json()["tasks"]) == 0

    # Verify tasks are indeed terminated (check status would be 404)
    status_response1 = client.get("/check_task_status/task_cleanup_1")
    assert status_response1.status_code == 404
    status_response2 = client.get("/check_task_status/task_cleanup_2")
    assert status_response2.status_code == 404


@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
async def test_create_task_process_creation_exception(mock_create_subprocess_exec: AsyncMock, client: TestClient):
    mock_create_subprocess_exec.side_effect = Exception("Process creation failed")
    task_cmd = ["echo", "hello"]
    task_uuid = "test_task_exception"
    response = client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    assert response.status_code == 500
    assert "Process creation failed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_gracefully_terminate_already_terminated(client: TestClient):
    # This test is a bit indirect as gracefully_terminate is not an endpoint
    # We test its behavior via the close_task endpoint for a quickly finishing task
    task_cmd = ["echo", "quick_finish"]
    task_uuid = "test_quick_finish"

    # Create and let it finish
    client.post("/create_task", json={"cmd": task_cmd, "uuid": task_uuid})
    await asyncio.sleep(0.2)  # Ensure it's completed

    status_resp = client.get(f"/check_task_status/{task_uuid}")
    assert status_resp.json()["status"] == "completed"

    close_response = client.post(f"/close_task/{task_uuid}")
    assert close_response.status_code == 200
