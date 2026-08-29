"""Internal support for sync and async user adapters."""

from __future__ import annotations

from inspect import isawaitable
from typing import Any


async def resolve(value: Any) -> Any:
    return await value if isawaitable(value) else value

