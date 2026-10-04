# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for the experimental read-only DuckDB database backend.

The tests need the DuckDB 'spatial' extension to be installed in the
default extension directory.
"""
import functools
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

import sqlalchemy as sa

import nominatim_api as napi
from nominatim_api.sql.duckdb_layout import LAYOUT_VERSION, LAYOUT_VERSION_PROPERTY

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('duckdb_engine')


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """ Static file handler with support for single byte ranges,
        as needed by the DuckDB httpfs extension.
    """
    def log_message(self, *args):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return None
        size = os.path.getsize(path)
        match = re.fullmatch(r'bytes=(\d+)-(\d*)', self.headers.get('Range', ''))
        fd = open(path, 'rb')
        if match:
            start = int(match[1])
            end = min(int(match[2]) if match[2] else size - 1, size - 1)
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            fd.seek(start)
            length = end - start + 1
        else:
            self.send_response(200)
            length = size
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        self._remaining = length
        return fd

    def copyfile(self, source, outputfile):
        while self._remaining > 0:
            chunk = source.read(min(65536, self._remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self._remaining -= len(chunk)


@pytest.fixture
def http_dir(tmp_path):
    """ Directory served over HTTP. Yields (directory, base URL).
        Skips when the DuckDB httpfs extension is not installed.
    """
    try:
        duckdb.connect(config={'autoinstall_known_extensions': False})\
              .execute('LOAD httpfs')
    except duckdb.Error:
        pytest.skip('DuckDB extension httpfs is not installed')

    directory = tmp_path / 'www'
    directory.mkdir()
    server = ThreadingHTTPServer(('127.0.0.1', 0),
                                 functools.partial(RangeRequestHandler,
                                                   directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield directory, f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()


def make_nominatim_duckdb(path, version=str(LAYOUT_VERSION)):
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE x AS SELECT * FROM (VALUES (42, 'a'), (43, 'b')) t(v, s)")
        conn.execute('CREATE TABLE nominatim_properties (property TEXT, value TEXT)')
        conn.execute("INSERT INTO nominatim_properties VALUES (?, ?)",
                     (LAYOUT_VERSION_PROPERTY, version))


@pytest.fixture
def duckdb_file(tmp_path):
    db = tmp_path / 'test_nominatim.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE x AS SELECT * FROM (VALUES (42, 'a'), (43, 'b')) t(v, s)")
    return db


@pytest.mark.asyncio
async def test_duckdb_dsn_opens_database(duckdb_file):
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}'}) as api:
        async with api.begin() as conn:
            assert conn.connection.dialect.name == 'duckdb'
            assert conn.connection.dialect.is_async
            assert await conn.scalar(sa.text('SELECT v FROM x WHERE s = :s'),
                                     {'s': 'a'}) == 42

            result = await conn.execute(sa.text('SELECT v, s FROM x ORDER BY v'))
            assert [tuple(r) for r in result] == [(42, 'a'), (43, 'b')]

        async with api.begin() as conn:  # pooled connection is reusable
            assert await conn.scalar(sa.text('SELECT count(*) FROM x')) == 2


@pytest.mark.asyncio
async def test_duckdb_is_read_only(duckdb_file):
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}'}) as api:
        with pytest.raises(sa.exc.DBAPIError, match='read-only'):
            async with api.begin() as conn:
                await conn.execute(sa.text('INSERT INTO x VALUES (1, :s)'), {'s': 'c'})


@pytest.mark.asyncio
async def test_duckdb_spatial_loaded(duckdb_file):
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}'}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT ST_AsText(ST_Point(1, 2))')) \
                == 'POINT (1 2)'
            assert await conn.scalar(sa.text(
                "SELECT current_setting('autoinstall_known_extensions')")) is False
            assert await conn.scalar(sa.text(
                "SELECT current_setting('autoload_known_extensions')")) is False


@pytest.mark.asyncio
async def test_duckdb_extension_dir_is_used(duckdb_file, tmp_path):
    extdir = tmp_path / 'empty_ext'
    extdir.mkdir()
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}',
                     'NOMINATIM_DUCKDB_EXTENSION_DIR': str(extdir)}) as api:
        # spatial is not available in the empty directory and must not be downloaded
        with pytest.raises(sa.exc.DBAPIError, match='spatial'):
            async with api.begin():
                pass

    assert not any(extdir.iterdir())


def test_duckdb_sync_api_status(duckdb_file):
    with duckdb.connect(str(duckdb_file)) as conn:
        conn.execute('CREATE TABLE import_status (lastimportdate TIMESTAMPTZ)')
        conn.execute("INSERT INTO import_status VALUES ('2022-12-07 14:14:46+00')")
        conn.execute('CREATE TABLE nominatim_properties (property TEXT, value TEXT)')
        conn.execute("INSERT INTO nominatim_properties VALUES ('database_version', '5.3.99-0')")
        conn.execute("INSERT INTO nominatim_properties VALUES (?, ?)",
                     (LAYOUT_VERSION_PROPERTY, str(LAYOUT_VERSION)))

    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}'}) as api:
        result = api.status()

    assert result.status == 0
    assert result.database_version == '5.3.99-0'
    assert result.data_updated.isoformat() == '2022-12-07T14:14:46+00:00'


def test_duckdb_missing_file_raises(tmp_path):
    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN':
                     f'duckdb:dbname={tmp_path / "no.duckdb"}'}) as api:
        with pytest.raises(napi.UsageError, match='DuckDB'):
            api.status()


@pytest.mark.parametrize('version', [None, str(LAYOUT_VERSION - 1), str(LAYOUT_VERSION + 1)])
def test_duckdb_layout_version_mismatch_raises(duckdb_file, version):
    with duckdb.connect(str(duckdb_file)) as conn:
        conn.execute('CREATE TABLE nominatim_properties (property TEXT, value TEXT)')
        if version is not None:
            conn.execute("INSERT INTO nominatim_properties VALUES (?, ?)",
                         (LAYOUT_VERSION_PROPERTY, version))

    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}'}) as api:
        with pytest.raises(napi.UsageError, match='nominatim convert'):
            api.status()


@pytest.mark.asyncio
async def test_duckdb_remote_file_over_http(http_dir):
    directory, url = http_dir
    make_nominatim_duckdb(directory / 'remote.duckdb')

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={url}/remote.duckdb'}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT v FROM x WHERE s = :s'),
                                     {'s': 'b'}) == 43
            assert await conn.scalar(sa.text('SELECT ST_AsText(ST_Point(1, 2))')) \
                == 'POINT (1 2)'
            assert await conn.scalar(sa.text(
                "SELECT current_setting('validate_external_file_cache')")) == 'NO_VALIDATION'
            assert await conn.scalar(sa.text(
                "SELECT current_setting('parquet_metadata_cache')")) is True


@pytest.mark.asyncio
async def test_duckdb_remote_file_is_read_only(http_dir):
    directory, url = http_dir
    make_nominatim_duckdb(directory / 'remote.duckdb')

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={url}/remote.duckdb'}) as api:
        with pytest.raises(sa.exc.DBAPIError, match='read-only|read only'):
            async with api.begin() as conn:
                await conn.execute(sa.text('INSERT INTO x VALUES (1, :s)'), {'s': 'c'})


@pytest.mark.asyncio
async def test_duckdb_remote_connections_share_instance(http_dir):
    directory, url = http_dir
    make_nominatim_duckdb(directory / 'remote.duckdb')

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={url}/remote.duckdb'}) as api:
        async with api.begin() as conn1, api.begin() as conn2:
            # attached databases belong to the instance, not to a connection
            await conn1.execute(sa.text("ATTACH ':memory:' AS probe"))
            assert await conn2.scalar(sa.text(
                "SELECT count(*) FROM duckdb_databases() WHERE database_name = 'probe'")) == 1
            assert await conn2.scalar(sa.text('SELECT count(*) FROM x')) == 2


def test_duckdb_remote_layout_version_mismatch_raises(http_dir):
    directory, url = http_dir
    make_nominatim_duckdb(directory / 'remote.duckdb', version='0')

    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={url}/remote.duckdb'}) as api:
        with pytest.raises(napi.UsageError, match='nominatim convert'):
            api.status()


@pytest.mark.asyncio
async def test_duckdb_init_sql_is_run(duckdb_file):
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}',
                     'NOMINATIM_DUCKDB_INIT_SQL': "SET threads = 1; SET memory_limit = '512MB';"
                     }) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text("SELECT current_setting('threads')")) == 1
