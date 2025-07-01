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
"""A simple task management server that can be used to manage the execution of tasks."""

import asyncio
import logging
import argparse
import multiprocessing
from contextlib import asynccontextmanager
from dataclasses import dataclass
from asyncio.subprocess import Process

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from l0.utils import get_num_processes

logger = logging.getLogger(__name__)

app = FastAPI(title="Task Management Server")


@dataclass
class TaskState:
    process: Process
    monitor: asyncio.Task
    status: str = "running"
    output: str = ""
    error: str = ""


# Data models
class TaskCreate(BaseModel):
    cmd: list[str]
    uuid: str


class TaskStatus(BaseModel):
    status: str
    output: str | None = None
    error: str | None = None


task_registry: dict[str, TaskState] = {}

MAX_CONCURRENT_TASKS = get_num_processes(None, 4) * 8
task_semaphore: asyncio.Semaphore | None = None
state_modification_lock: asyncio.Lock | None = None


async def monitor_process(task_uuid: str, process: asyncio.subprocess.Process):
    """Monitor a single process."""
    try:
        stdout, stderr = await process.communicate()
        task_state = task_registry[task_uuid]
        if process.returncode == 0:
            task_state.status = "completed"
            task_state.output = stdout.decode() if stdout else ""
            logger.debug(f"task_executing output: {task_state.output}")
        else:
            task_state.status = "failed"
            task_state.error = stderr.decode() if stderr else ""
            logger.debug(f"task_executing error: {task_state.error}")
    except asyncio.CancelledError:
        logger.info(f"Monitor task for {task_uuid} was cancelled")
    except Exception as e:
        if task_uuid in task_registry:
            task_state = task_registry[task_uuid]
            task_state.status = "failed"
            task_state.error = str(e)
    finally:
        if task_semaphore:
            task_semaphore.release()
            logger.info(f"Released semaphore for task {task_uuid}. Available slots: {task_semaphore._value}")


async def cleanup():
    """Cleanup all running tasks and monitor tasks."""
    task_uuids = list(task_registry.keys())
    for uuid in task_uuids:
        await close_task(uuid)
    task_registry.clear()
    logger.info("Cleanup complete.")


async def gracefully_terminate(proc: Process, timeout_graceful: float = 1.0, timeout_force: float = 1.0):
    """Attempts to terminate the process gracefully, then forcefully kills it if necessary.

    Args:
        proc: The process to terminate
        timeout_graceful: Timeout in seconds for graceful termination (SIGTERM)
        timeout_force: Timeout in seconds for forced termination (SIGKILL)
    """
    if proc.returncode is not None:
        logger.debug(f"Process {proc.pid} already terminated.")
        return

    try:
        logger.info(f"Sending SIGTERM to process {proc.pid}.")
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=timeout_graceful)
        if proc.returncode is not None:
            logger.info(f"Process {proc.pid} terminated gracefully after SIGTERM.")
            return
    except asyncio.TimeoutError:
        logger.warning(
            f"Process {proc.pid} did not terminate via SIGTERM within {timeout_graceful}s. Attempting SIGKILL."
        )
    except Exception as e:
        logger.warning(f"Exception during SIGTERM handling for process {proc.pid}: {e}. Attempting SIGKILL.")

    if proc.returncode is None:
        logger.warning(f"Sending SIGKILL to process {proc.pid}.")
        try:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=timeout_force)
            if proc.returncode is not None:
                logger.info(f"Process {proc.pid} terminated after SIGKILL.")
            else:
                logger.error(f"Process {proc.pid} still running after SIGKILL and wait. This is unexpected.")
        except asyncio.TimeoutError:
            logger.error(
                f"Process {proc.pid} failed to die via SIGKILL within {timeout_force}s. It might be a zombie or in an uninterruptible state."
            )
        except Exception as e:
            logger.error(f"Exception during SIGKILL handling for process {proc.pid}: {e}.")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global task_semaphore, state_modification_lock, logger

    uvicorn_logger = logging.getLogger("uvicorn.error")
    logger.handlers = uvicorn_logger.handlers
    logger.setLevel(uvicorn_logger.level)

    task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    state_modification_lock = asyncio.Lock()

    logger.info(f"Task server started with max {MAX_CONCURRENT_TASKS} concurrent tasks")

    yield

    await cleanup()


app = FastAPI(title="Task Management Server", lifespan=lifespan)


@app.post("/create_task")
async def create_task(task: TaskCreate):
    try:
        if task.uuid in task_registry and task_registry[task.uuid].status == "running":
            raise ValueError(f"Task with UUID {task.uuid} is already running.")

        await task_semaphore.acquire()
        logger.info(f"Acquired semaphore for task {task.uuid}. Available slots: {task_semaphore._value}")

        try:
            async with state_modification_lock:
                process = await asyncio.create_subprocess_exec(
                    *task.cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )

                monitor_task = asyncio.create_task(monitor_process(task.uuid, process), name=f"monitor_{task.uuid}")

                task_registry[task.uuid] = TaskState(process=process, monitor=monitor_task)
                await asyncio.sleep(0.01)
            return {"status": "OK"}
        except Exception:
            task_semaphore.release()
            logger.info(
                f"Released semaphore for task {task.uuid} due to exception. Available slots: {task_semaphore._value}"
            )
            raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/check_task_status/{task_uuid}")
async def check_task_status(task_uuid: str):
    if task_uuid not in task_registry:
        raise HTTPException(status_code=404, detail="Task not found")

    task_state = task_registry[task_uuid]
    logger.info(
        f"Checking status for task {task_uuid}: {task_state.status} with output: {task_state.output} and error: {task_state.error}"
    )
    return {"status": task_state.status, "output": task_state.output, "error": task_state.error}


@app.post("/close_task/{task_uuid}")
async def close_task(task_uuid: str):
    if task_uuid not in task_registry:
        raise HTTPException(status_code=404, detail="Task not found")

    async with state_modification_lock:
        task_state = task_registry[task_uuid]
        task_registry.pop(task_uuid)

    if task_state.status == "running":
        try:
            process = task_state.process
            await gracefully_terminate(process)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    if task_state.monitor and not task_state.monitor.done():
        try:
            task_state.monitor.cancel()
            await task_state.monitor
        except Exception:
            logger.warning(f"Monitor task for {task_uuid} was cancelled or failed to complete.")

    return {"status": "OK"}


@app.post("/cleanup")
async def cleanup_endpoint():
    """Manually trigger cleanup of all tasks."""
    try:
        await cleanup()
        return {"status": "OK", "message": "All tasks cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/list_tasks")
async def list_tasks():
    return {"tasks": list(task_registry.keys())}


if __name__ == "__main__":
    import uvicorn

    multiprocessing.set_start_method(method="fork", force=True)

    parser = argparse.ArgumentParser(description="Task Management Server")
    parser.add_argument("--port", type=int, default=8000, help=f"Port to run the server on (default: {8000})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
