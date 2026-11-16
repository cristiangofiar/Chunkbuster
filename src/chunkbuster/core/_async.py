"""Query task lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def task_scope():
    """Own query tasks and drain them on errors or caller cancellation."""
    tasks = []

    def start(coroutine):
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    try:
        yield start
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
