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
