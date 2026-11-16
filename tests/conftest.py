import asyncio

import pytest


@pytest.fixture(params=[None, asyncio.eager_task_factory], ids=["default", "eager"])
async def task_factory(request):
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    loop.set_task_factory(request.param)
    try:
        yield
    finally:
        loop.set_task_factory(previous)
