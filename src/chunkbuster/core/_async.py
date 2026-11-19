"""Query task lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def task_scope():
    """Own query tasks and drain them on errors or caller cancellation."""
    tasks = []
    closing = False
    owner = asyncio.current_task()
    initial_cancellations = owner.cancelling()

    def start(coroutine):
        if closing or owner.cancelling() > initial_cancellations:
            coroutine.close()
            raise asyncio.CancelledError
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    try:
        yield start
    finally:
        closing = True
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
