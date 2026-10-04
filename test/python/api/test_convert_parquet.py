# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for exporting a DuckDB database to Parquet.
"""
import json
import sys

import pytest

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('pyarrow', minversion='24')

from nominatim_db.tools import convert_parquet  # noqa: E402
from nominatim_db.errors import UsageError  # noqa: E402


@pytest.fixture
def con():
    con = duckdb.connect()
    con.load_extension('spatial')
    # 3000 small points with three large polygons in between
    con.execute("""CREATE TABLE placex AS
                   SELECT i AS place_id,
                          CASE WHEN i IN (500, 1500, 2500)
                               THEN ST_Buffer(ST_Point(i, 0), 10, 2000)
                               ELSE ST_Point(i, 0) END AS geometry,
                          '{"name": "東京"}'::JSON AS name,
                          [1, 2]::INTEGER[] AS vector,
                          ['osm.amenity.cafe']::VARCHAR[] AS categories,
                          i % 2 = 0 AS flag
                     FROM range(3000) t(i) ORDER BY i""")
    yield con
    con.close()


def rowgroups(path):
    return duckdb.connect().execute(
        f"""SELECT row_group_id, max(row_group_num_rows)
              FROM parquet_metadata('{path}') GROUP BY 1 ORDER BY 1""").fetchall()


def test_export_rows_only(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    groups = convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out,
                                          None, max_rows=1000)
    assert groups == 3
    assert [n for _, n in rowgroups(out)] == [1000, 1000, 1000]


def test_export_byte_budget_isolates_large_rows(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    budget = ('octet_length(ST_AsWKB(geometry))', 50000)
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out, budget)

    reader = duckdb.connect()
    big_groups = reader.execute(
        f"""SELECT DISTINCT row_group_id FROM parquet_metadata('{out}') m
             WHERE path_in_schema = 'place_id'
               AND stats_min::INT <= 2500 AND stats_max::INT >= 2500""").fetchall()
    assert len(big_groups) == 1
    sizes = dict(rowgroups(out))
    assert sizes[big_groups[0][0]] < 3000  # the large polygon closes its row group
    assert len(sizes) >= 4


def test_export_keeps_row_order(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out,
                                 ('octet_length(ST_AsWKB(geometry))', 50000))
    rows = duckdb.connect().execute(
        f"SELECT file_row_number, place_id FROM read_parquet('{out}', file_row_number = true)"
    ).fetchall()
    assert all(rn == pid for rn, pid in rows)


def test_export_keeps_types(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out, None)

    reader = duckdb.connect()
    reader.load_extension('spatial')
    types = {r[0]: r[1] for r in reader.execute(f"DESCRIBE SELECT * FROM '{out}'").fetchall()}
    assert types['name'] == 'JSON'
    assert types['vector'] == 'INTEGER[]'
    assert types['categories'] == 'VARCHAR[]'
    assert types['flag'] == 'BOOLEAN'  # not TINYINT (arrow.bool8)
    assert types['geometry'].startswith('GEOMETRY')
    assert reader.execute(f"SELECT name->>'name' FROM '{out}' LIMIT 1").fetchone()[0] == '東京'


def test_export_writes_geoparquet_metadata(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out, None)

    meta = duckdb.connect().execute(
        f"SELECT decode(value) FROM parquet_kv_metadata('{out}') WHERE key = 'geo'"
    ).fetchone()[0]
    geo = json.loads(meta)
    assert geo['version'] == '1.0.0'
    assert geo['primary_column'] == 'geometry'
    assert geo['columns']['geometry']['encoding'] == 'WKB'
    assert set(geo['columns']['geometry']['geometry_types']) == {'Point', 'Polygon'}
    assert geo['columns']['geometry']['bbox'] == pytest.approx([0.0, -10.0, 2999.0, 10.0],
                                                               abs=0.01)


def test_export_bloom_filters(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex', out, None,
                                 max_rows=1000, bloom_filters=('place_id', ))

    reader = duckdb.connect()
    assert reader.execute(
        f"""SELECT DISTINCT path_in_schema FROM parquet_metadata('{out}')
             WHERE bloom_filter_offset IS NOT NULL""").fetchall() == [('place_id', )]
    # place_id 1500 is in the second row group only
    assert reader.execute(
        f"""SELECT row_group_id, bloom_filter_excludes
              FROM parquet_bloom_probe('{out}', 'place_id', 1500) ORDER BY 1""").fetchall() \
        == [(0, True), (1, False), (2, True)]


def test_import_pyarrow_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    with pytest.raises(UsageError, match='pyarrow'):
        convert_parquet.import_pyarrow()


def test_import_pyarrow_too_old(monkeypatch):
    import pyarrow
    monkeypatch.setattr(pyarrow, '__version__', '23.0.0')
    with pytest.raises(UsageError, match='pyarrow>=24'):
        convert_parquet.import_pyarrow()
