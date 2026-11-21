import asyncio

import numpy as np
import pytest

from rageval.serve.batcher import MicroBatcher


async def test_batches_concurrent_requests_and_preserves_order():
    sizes = []

    def fn(items):
        sizes.append(len(items))
        return np.array([x * 10 for x in items], dtype=np.float32)

    b = MicroBatcher(fn, "t", max_items=64, max_wait_ms=20)
    outs = await asyncio.gather(*(b.submit([i, i + 100]) for i in range(10)))
    for i, o in enumerate(outs):
        assert o.tolist() == [i * 10, (i + 100) * 10]
    assert max(sizes) > 2  # requests were merged
    b.close()


async def test_2d_outputs_are_sliced_per_request():
    b = MicroBatcher(lambda items: np.stack([np.full(3, x, dtype=np.float32) for x in items]), "t2")
    a, c = await asyncio.gather(b.submit([1]), b.submit([2, 3]))
    assert a.shape == (1, 3) and c.shape == (2, 3) and c[1, 0] == 3
    b.close()


async def test_errors_propagate():
    def boom(items):
        raise RuntimeError("gpu fell over")

    b = MicroBatcher(boom, "t3")
    with pytest.raises(RuntimeError):
        await b.submit([1])
    b.close()
