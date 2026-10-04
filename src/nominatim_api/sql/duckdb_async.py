# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Async adaptor for DuckDB, so that it can be used with SQLAlchemy's
asyncio extension.

DuckDB is an in-process database with a synchronous Python API only.
SQLAlchemy's async engine requires a dialect with `is_async` set, whose
DBAPI connection follows the protocol of the generic adaptor classes in
`sqlalchemy.connectors.asyncio`. These classes expect an asyncio-style
driver (awaitable execute/fetch/commit) underneath and bridge it to the
synchronous DBAPI interface through the greenlet mechanism.

This module therefore wraps the synchronous connection of the
`duckdb_engine` dialect into a thin shim with coroutine methods and lets
the standard SQLAlchemy adaptor classes do the rest. For local database
files the coroutines run the DuckDB calls directly, i.e. a query blocks
the event loop while it executes. This is acceptable for an in-process
database and avoids the overhead of handing every call over to a worker
thread. Remote and Parquet sources may wait for seconds on network I/O,
so their queries and fetches are run in a worker thread instead (connect
argument `nominatim_offload`). When such a call is cancelled, for example
by the query timeout, the DuckDB query is interrupted as well.
"""
from typing import Any, Callable, Optional, Sequence, Type
import asyncio
import contextvars
import functools

import sqlalchemy as sa
from sqlalchemy.connectors.asyncio import AsyncAdapt_dbapi_connection
from sqlalchemy.dialects.postgresql.psycopg2 import PGExecutionContext_psycopg2
from duckdb_engine import Dialect as DuckDBDialect


async def _run(offload: bool, interrupt: Optional[Callable[[], None]],
               func: Callable[..., Any], *args: Any) -> Any:
    """ Run the synchronous DuckDB call `func`, in a worker thread
        when `offload` is set.

        When the waiting coroutine is cancelled (for example by the
        query timeout), the DuckDB query is stopped with `interrupt` and
        the cancellation is only passed on once the worker thread has
        returned. Otherwise the connection could go back to the pool
        while the thread is still using it.
    """
    if not offload:
        return func(*args)

    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    future = loop.run_in_executor(None, functools.partial(ctx.run, func, *args))
    try:
        # Unlike awaiting the future directly, waiting does not cancel it.
        await asyncio.wait((future, ))
    except asyncio.CancelledError:
        while not future.done():
            if interrupt is not None:
                # Repeated, in case the query had not started yet. An error
                # must not end the wait for the worker thread.
                try:
                    interrupt()
                except Exception:
                    pass
            try:
                await asyncio.wait((future, ), timeout=0.1)
            except asyncio.CancelledError:
                pass
        if not future.cancelled():
            future.exception()  # mark a resulting error as retrieved
        raise

    return future.result()


class _AsyncDuckDBCursor:
    """ Coroutine-style facade for a duckdb_engine cursor.
    """

    def __init__(self, cursor: Any, offload: bool = False,
                 interrupt: Optional[Callable[[], None]] = None) -> None:
        self._cursor = cursor
        self._offload = offload
        self._interrupt = interrupt

    async def __aenter__(self) -> '_AsyncDuckDBCursor':
        return self

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    @property
    def arraysize(self) -> int:
        return 1

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        pass  # rows are always fully buffered by the adaptor

    async def execute(self, operation: Any, parameters: Any = None) -> Any:
        return await _run(self._offload, self._interrupt,
                          self._cursor.execute, operation, parameters)

    async def executemany(self, operation: Any, seq_of_parameters: Any) -> Any:
        return await _run(self._offload, self._interrupt,
                          self._cursor.executemany, operation, seq_of_parameters)

    async def fetchone(self) -> Optional[Any]:
        return await _run(self._offload, self._interrupt, self._cursor.fetchone)

    async def fetchmany(self, size: Optional[int] = None) -> Sequence[Any]:
        return await _run(self._offload,  # type: ignore[no-any-return]
                          self._interrupt, self._cursor.fetchmany, size)

    async def fetchall(self) -> Sequence[Any]:
        return await _run(self._offload,  # type: ignore[no-any-return]
                          self._interrupt, self._cursor.fetchall)

    async def nextset(self) -> Optional[bool]:
        return None

    async def setinputsizes(self, *_: Any) -> None:
        pass

    async def close(self) -> None:
        self._cursor.close()


class _AsyncDuckDBConnection:
    """ Coroutine-style facade for a duckdb_engine connection.
        Other attributes are passed through to the synchronous connection.
    """

    def __init__(self, connection: Any, offload: bool = False) -> None:
        self._connection = connection
        self._offload = offload

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _AsyncDuckDBCursor(self._connection.cursor(*args, **kwargs), self._offload,
                                  self._connection.interrupt if self._offload else None)

    async def begin(self) -> None:
        self._connection.begin()

    async def commit(self) -> None:
        self._connection.commit()

    async def rollback(self) -> None:
        self._connection.rollback()

    async def close(self) -> None:
        self._connection.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class AsyncAdapt_duckdb_connection(AsyncAdapt_dbapi_connection):
    """ DBAPI connection as handed out by the SQLAlchemy pool.
        duckdb_engine starts transactions explicitly with `begin()`,
        which the generic adaptor does not provide.
    """
    __slots__ = ()

    def begin(self) -> None:
        try:
            self.await_(self._connection.begin())
        except Exception as error:
            self._handle_exception(error)


class _AsyncDuckDBExecutionContext(PGExecutionContext_psycopg2):
    """ Execution context of duckdb_engine (inherited from psycopg2),
        without the forwarding of PostgreSQL notices, which would
        require access to the driver connection through the cursor.
    """

    def post_exec(self) -> None:
        pass


class AsyncDuckDBDialect(DuckDBDialect):  # type: ignore[misc,unused-ignore]
    """ duckdb_engine dialect usable with `create_async_engine()`.
        Register as 'duckdb.aioduckdb' and use the URL `duckdb+aioduckdb://`.
    """
    driver = 'aioduckdb'
    is_async = True
    supports_statement_cache = True
    execution_ctx_cls = _AsyncDuckDBExecutionContext

    @classmethod
    def get_pool_class(cls, url: sa.engine.URL) -> Type[sa.pool.Pool]:
        return sa.pool.AsyncAdaptedQueuePool

    def connect_sync(self, *cargs: Any, **cparams: Any) -> Any:
        """ Open a plain synchronous duckdb_engine connection with the
            same arguments and so the same DuckDB configuration as the
            connections of the pool.
        """
        cparams.pop('nominatim_offload', None)
        return super().connect(*cargs, **cparams)

    def connect(self, *cargs: Any, **cparams: Any) -> Any:
        offload = bool(cparams.pop('nominatim_offload', False))
        return AsyncAdapt_duckdb_connection(
                    self.loaded_dbapi,
                    _AsyncDuckDBConnection(super().connect(*cargs, **cparams), offload))
