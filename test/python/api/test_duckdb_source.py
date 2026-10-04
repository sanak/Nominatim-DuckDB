# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for the parsing of DuckDB data source names.
"""
import pytest

from nominatim_api.errors import UsageError
from nominatim_api.sql.duckdb_source import DuckDBSource, parse_dsn, is_url


@pytest.mark.parametrize('dsn,kind,location', [
    ('dbname=/srv/x.duckdb', 'file', '/srv/x.duckdb'),
    ('dbname=rel.duckdb', 'file', 'rel.duckdb'),
    ('dbname=s3://bucket/x.duckdb', 'remote', 's3://bucket/x.duckdb'),
    ('dbname=https://example.com/x.duckdb', 'remote', 'https://example.com/x.duckdb'),
    ('parquet=s3://bucket/japan/', 'parquet', 's3://bucket/japan/'),
    ('parquet=/data/japan', 'parquet', '/data/japan'),
])
def test_parse_dsn(dsn, kind, location):
    assert parse_dsn(dsn) == DuckDBSource(kind, location)


def test_parse_dsn_without_dbname_is_empty_file():
    assert parse_dsn('') == DuckDBSource('file', '')


def test_parse_dsn_rejects_dbname_and_parquet():
    with pytest.raises(UsageError, match='either'):
        parse_dsn('dbname=x.duckdb;parquet=/data')


def test_is_url():
    assert is_url('s3://b/k')
    assert is_url('http://127.0.0.1:8000/x')
    assert not is_url('/srv/x.duckdb')


def test_instance_name_is_stable_and_specific():
    src = DuckDBSource('remote', 's3://b/x.duckdb')
    name = src.instance_name('')

    assert name.startswith(':memory:nominatim_')
    assert name == DuckDBSource('remote', 's3://b/x.duckdb').instance_name('')
    assert name != DuckDBSource('remote', 's3://b/y.duckdb').instance_name('')
    assert name != src.instance_name('/srv/duckdb-extensions')
    assert name != DuckDBSource('parquet', 's3://b/x.duckdb').instance_name('')


@pytest.mark.parametrize('location', ['s3://b/japan', 's3://b/japan/'])
def test_parquet_file(location):
    assert DuckDBSource('parquet', location).parquet_file('placex') \
        == 's3://b/japan/placex.parquet'
