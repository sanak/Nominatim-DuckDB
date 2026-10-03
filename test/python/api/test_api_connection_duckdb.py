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
import pytest

import sqlalchemy as sa

import nominatim_api as napi
from nominatim_api.sql.duckdb_layout import LAYOUT_VERSION, LAYOUT_VERSION_PROPERTY

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('duckdb_engine')


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
