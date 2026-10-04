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
import asyncio
import functools
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

import sqlalchemy as sa

import nominatim_api as napi
from nominatim_api.sql.duckdb_layout import (LAYOUT_VERSION, LAYOUT_VERSION_PROPERTY,
                                             PARQUET_TABLES_PROPERTY, STORAGE_FORMAT_PROPERTY)
from nominatim_api.sql.duckdb_source import parse_dsn

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


@pytest.fixture
def httpfs_installed():
    try:
        duckdb.connect(config={'autoinstall_known_extensions': False})\
              .execute('LOAD httpfs')
    except duckdb.Error:
        pytest.skip('DuckDB extension httpfs is not installed')


S3_SECRET_SQL = ("CREATE SECRET IF NOT EXISTS nominatim_s3"
                 " (TYPE s3, KEY_ID 'k', SECRET 's', REGION 'us-east-1')")


@pytest.mark.asyncio
async def test_duckdb_init_sql_with_s3_secret_on_local_file(duckdb_file, httpfs_installed):
    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={duckdb_file}',
                     'NOMINATIM_DUCKDB_INIT_SQL': S3_SECRET_SQL}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT count(*) FROM x')) == 2


def make_parquet_dir(directory, tables='nominatim_properties,x', version=str(LAYOUT_VERSION)):
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE x AS SELECT * FROM (VALUES (42, 'a'), (43, 'b')) t(v, s)")
        conn.execute(f"COPY x TO '{directory / 'x.parquet'}' (FORMAT parquet)")
        conn.execute('CREATE TABLE p (property TEXT, value TEXT)')
        conn.execute('INSERT INTO p VALUES (?, ?), (?, ?)',
                     (LAYOUT_VERSION_PROPERTY, version, STORAGE_FORMAT_PROPERTY, 'parquet'))
        if tables is not None:
            conn.execute('INSERT INTO p VALUES (?, ?)', (PARQUET_TABLES_PROPERTY, tables))
        conn.execute(f"COPY p TO '{directory / 'nominatim_properties.parquet'}' (FORMAT parquet)")


@pytest.mark.asyncio
async def test_duckdb_parquet_local_directory(tmp_path):
    make_parquet_dir(tmp_path)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        async with api.begin() as conn:
            result = await conn.execute(sa.text('SELECT rowid, v, s FROM x ORDER BY rowid'))
            assert [tuple(r) for r in result] == [(0, 42, 'a'), (1, 43, 'b')]
            assert await conn.scalar(sa.text('SELECT v FROM x WHERE rowid IN (1)')) == 43


@pytest.mark.asyncio
async def test_duckdb_init_sql_with_s3_secret_on_local_parquet(tmp_path, httpfs_installed):
    make_parquet_dir(tmp_path)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}',
                     'NOMINATIM_DUCKDB_INIT_SQL': S3_SECRET_SQL}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT count(*) FROM x')) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['dbname', 'parquet'])
async def test_duckdb_init_sql_with_s3_secret_on_two_connections(tmp_path, duckdb_file,
                                                                 httpfs_installed, kind):
    # The init SQL runs on every new connection, while secrets belong to
    # the DuckDB instance, which the connections share.
    if kind == 'parquet':
        make_parquet_dir(tmp_path)
        dsn = f'duckdb:parquet={tmp_path}'
    else:
        dsn = f'duckdb:dbname={duckdb_file}'

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': dsn,
                     'NOMINATIM_DUCKDB_INIT_SQL': S3_SECRET_SQL}) as api:
        async with api.begin() as conn1, api.begin() as conn2:
            assert await conn1.scalar(sa.text('SELECT count(*) FROM x')) == 2
            assert await conn2.scalar(sa.text('SELECT count(*) FROM x')) == 2


@pytest.mark.asyncio
async def test_duckdb_parquet_over_http(http_dir):
    directory, url = http_dir
    make_parquet_dir(directory)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={url}/'}) as api:
        async with api.begin() as conn1, api.begin() as conn2:
            assert await conn1.scalar(sa.text('SELECT count(*) FROM x')) == 2
            assert await conn2.scalar(sa.text('SELECT max(v) FROM x')) == 43


@pytest.mark.asyncio
async def test_duckdb_parquet_only_listed_tables(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties')  # x.parquet exists, not listed

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'x'")) == 0


def test_duckdb_parquet_without_table_list_raises(tmp_path):
    make_parquet_dir(tmp_path, tables=None)

    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        with pytest.raises(napi.UsageError, match=PARQUET_TABLES_PROPERTY):
            api.status()


def test_duckdb_parquet_without_properties_raises(tmp_path):
    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        with pytest.raises(napi.UsageError, match='Parquet'):
            api.status()


def test_duckdb_parquet_missing_directory_raises(tmp_path):
    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path / "nope"}'}) as api:
        with pytest.raises(napi.UsageError, match='does not exist'):
            api.status()


def test_duckdb_parquet_layout_version_mismatch_raises(tmp_path):
    make_parquet_dir(tmp_path, version='0')

    with napi.NominatimAPI(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        with pytest.raises(napi.UsageError, match='nominatim convert'):
            api.status()


def make_search_name_file(directory):
    with duckdb.connect() as conn:
        conn.execute(f"""COPY (SELECT * FROM (VALUES (7, [1]), (5, [2]), (9, [3]))
                                         t(place_id, name_vector))
                         TO '{directory / 'search_name.parquet'}' (FORMAT parquet)""")


@pytest.mark.asyncio
async def test_duckdb_parquet_loads_search_name_into_memory(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties,x,search_name')
    make_search_name_file(tmp_path)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text(
                "SELECT table_type FROM information_schema.tables"
                " WHERE table_name = 'search_name'")) == 'BASE TABLE'
            assert await conn.scalar(sa.text(
                "SELECT table_type FROM information_schema.tables"
                " WHERE table_name = 'x'")) == 'VIEW'
            # rowid is the Parquet row number, like the row id of the database file
            result = await conn.execute(sa.text(
                'SELECT rowid, place_id FROM search_name WHERE rowid IN'
                ' (SELECT rowid FROM search_name WHERE place_id IN (5, 9)) ORDER BY rowid'))
            assert [tuple(r) for r in result] == [(1, 5), (2, 9)]


@pytest.mark.asyncio
async def test_duckdb_parquet_loads_search_name_once(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties,x,search_name')
    make_search_name_file(tmp_path)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'}) as api:
        async with api.begin() as conn1:
            assert await conn1.scalar(sa.text('SELECT count(*) FROM search_name')) == 3
            (tmp_path / 'search_name.parquet').unlink()
            # A new connection of the pool uses the loaded table of the
            # shared instance and does not read the file again.
            async with api.begin() as conn2:
                assert await conn2.scalar(sa.text('SELECT count(*) FROM search_name')) == 3


@pytest.mark.asyncio
async def test_duckdb_parquet_keeps_search_name_without_pool(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties,x,search_name')
    make_search_name_file(tmp_path)

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}',
                     'NOMINATIM_API_POOL_SIZE': '0'}) as api:
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT count(*) FROM search_name')) == 3
        (tmp_path / 'search_name.parquet').unlink()
        # Without a pool, every request opens a new connection. The shared
        # instance stays open and the table is not read again.
        async with api.begin() as conn:
            assert await conn.scalar(sa.text('SELECT count(*) FROM search_name')) == 3


@pytest.mark.asyncio
async def test_duckdb_remote_keeps_instance_without_pool(http_dir):
    directory, url = http_dir
    make_nominatim_duckdb(directory / 'remote.duckdb')

    async with napi.NominatimAPIAsync(
            environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={url}/remote.duckdb',
                     'NOMINATIM_API_POOL_SIZE': '0'}) as api:
        async with api.begin() as conn:
            await conn.execute(sa.text("ATTACH ':memory:' AS probe"))
        async with api.begin() as conn:
            assert await conn.scalar(sa.text(
                "SELECT count(*) FROM duckdb_databases() WHERE database_name = 'probe'")) == 1
            assert await conn.scalar(sa.text('SELECT count(*) FROM x')) == 2


@pytest.mark.asyncio
async def test_duckdb_parquet_concurrent_first_connections(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties,x,search_name')
    make_search_name_file(tmp_path)

    async def _count(api):
        async with api.begin() as conn:
            return await conn.scalar(sa.text('SELECT count(*) FROM search_name'))

    for _ in range(5):
        async with napi.NominatimAPIAsync(
                environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}',
                         'NOMINATIM_API_POOL_SIZE': '0'}) as api:
            assert await asyncio.gather(*(_count(api) for _ in range(4))) == [3] * 4


@pytest.mark.asyncio
async def test_duckdb_parquet_close_releases_instance(tmp_path):
    make_parquet_dir(tmp_path, tables='nominatim_properties,x,search_name')
    make_search_name_file(tmp_path)
    instance = parse_dsn(f'parquet={tmp_path}').instance_name('')

    api = napi.NominatimAPIAsync(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:parquet={tmp_path}'})
    async with api.begin() as conn:
        assert await conn.scalar(sa.text('SELECT count(*) FROM search_name')) == 3
    await api.close()

    # A new connection with a different configuration would be refused
    # if the instance were still open.
    with duckdb.connect(instance) as conn:
        assert conn.execute('SELECT count(*) FROM duckdb_tables()').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM duckdb_views()'
                            ' WHERE NOT internal').fetchone()[0] == 0

    # The API can be used again after closing.
    async with api.begin() as conn:
        assert await conn.scalar(sa.text('SELECT count(*) FROM search_name')) == 3
    await api.close()
