import asyncio

import pytest

from chunkbuster.core._async import task_scope


@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_task_scope_drains_siblings_and_preserves_error_or_cancellation(
    cancel_caller,
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def blocked():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def fail():
        await started.wait()
        raise ValueError("binding failed")

    async def run():
        async with task_scope() as start:
            start(blocked())
            if cancel_caller:
                await asyncio.Event().wait()
            else:
                await start(fail())

    caller = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), timeout=2)
    if cancel_caller:
        caller.cancel()
    error = asyncio.CancelledError if cancel_caller else ValueError
    with pytest.raises(error):
        await asyncio.wait_for(caller, timeout=2)
    assert stopped.is_set()
