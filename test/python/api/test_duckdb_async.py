# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for the asyncio adaptor of DuckDB.
"""
import asyncio
import time

import pytest

pytest.importorskip('duckdb_engine')

from nominatim_api.sql.duckdb_async import _AsyncDuckDBCursor  # noqa: E402


class SlowCursor:
    def execute(self, *args):
        time.sleep(0.2)
        return self

    def fetchall(self):
        time.sleep(0.2)
        return [(1, )]


async def _ticks_during(coro):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker start
    result = await coro
    task.cancel()
    return ticks, result


@pytest.mark.asyncio
async def test_offloaded_execute_does_not_block_event_loop():
    ticks, _ = await _ticks_during(_AsyncDuckDBCursor(SlowCursor(), offload=True)
                                   .execute('SELECT 1'))
    assert ticks >= 5


@pytest.mark.asyncio
async def test_offloaded_fetch_does_not_block_event_loop():
    ticks, rows = await _ticks_during(_AsyncDuckDBCursor(SlowCursor(), offload=True)
                                      .fetchall())
    assert ticks >= 5
    assert rows == [(1, )]


@pytest.mark.asyncio
async def test_inline_execute_runs_on_event_loop():
    ticks, _ = await _ticks_during(_AsyncDuckDBCursor(SlowCursor()).execute('SELECT 1'))
    assert ticks == 0


@pytest.mark.asyncio
async def test_cancelled_offloaded_query_is_interrupted():
    duckdb = pytest.importorskip('duckdb')
    from duckdb_engine import ConnectionWrapper
    from nominatim_api.sql.duckdb_async import _AsyncDuckDBConnection

    conn = _AsyncDuckDBConnection(ConnectionWrapper(duckdb.connect()), offload=True)
    try:
        cursor = conn.cursor()
        with pytest.raises(asyncio.TimeoutError):
            # Runs for many seconds unless interrupted.
            await asyncio.wait_for(
                cursor.execute('SELECT count(*) FROM range(100_000_000_000)'), 0.2)

        start = time.monotonic()
        cursor = conn.cursor()
        await cursor.execute('SELECT 42')
        assert await cursor.fetchall() == [(42, )]
        assert time.monotonic() - start < 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_cancelled_offloaded_call_waits_when_interrupt_fails():
    finished = []

    class _Cursor:
        def execute(self, *_):
            time.sleep(0.3)
            finished.append(True)

    def _interrupt():
        raise RuntimeError('connection already closed')

    cursor = _AsyncDuckDBCursor(_Cursor(), offload=True, interrupt=_interrupt)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(cursor.execute('SELECT 1'), 0.05)
    # The cancellation is passed on only after the worker has returned.
    assert finished == [True]
