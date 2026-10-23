"""One process-wide inference slot shared by cached Mars and Earth requests.

Cancellation never releases the slot while the worker still owns GPU tensors.
The existing analysis-cache claim protocol continues to deduplicate requests;
this slot also serializes computations with different cache identities.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import torch

if not hasattr(torch, "_astraatmos_inference_slot"):
    torch._astraatmos_inference_slot = threading.BoundedSemaphore(1)

INFERENCE_SLOT = torch._astraatmos_inference_slot


async def run_inference_compute(compute):
    while not INFERENCE_SLOT.acquire(blocking=False):
        await asyncio.sleep(0.05)

    async def invoke():
        result = compute()
        return await result if inspect.isawaitable(result) else result

    work = asyncio.create_task(invoke())
    # The callback owns the release: even repeated cancellation cannot release
    # the slot before asyncio.to_thread has finished its underlying worker.
    def finished(task):
        INFERENCE_SLOT.release()
        if not task.cancelled():
            task.exception()  # Retrieve failures even if the HTTP caller left.

    work.add_done_callback(finished)
    return await asyncio.shield(work)
