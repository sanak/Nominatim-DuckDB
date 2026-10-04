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
    # no CRS: GEOMETRY('OGC:CRS84') cannot be cast to POINT_2D
    assert types['geometry'] == 'GEOMETRY'
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
    assert geo['columns']['geometry']['crs'] is None
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


def column_encodings(path):
    return {r[0]: (r[1], r[2]) for r in duckdb.connect().execute(
        f"""SELECT path_in_schema, any_value(compression), any_value(encodings)
              FROM parquet_metadata('{path}') GROUP BY 1""").fetchall()}


def test_export_default_writer_options(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT place_id FROM placex', out, None)

    compression, encodings = column_encodings(out)['place_id']
    assert compression == 'ZSTD'
    assert 'RLE_DICTIONARY' in encodings


def test_export_writer_options(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(
        con, 'placex', 'SELECT place_id, vector, name FROM placex', out, None,
        writer_options={'compression': 'lz4_raw', 'use_dictionary': ['name'],
                        'column_encoding': {'place_id': 'DELTA_BINARY_PACKED'}})

    columns = column_encodings(out)
    assert {c[0] for c in columns.values()} == {'LZ4_RAW'}
    assert 'DELTA_BINARY_PACKED' in columns['place_id'][1]
    assert 'RLE_DICTIONARY' not in columns['vector, list, element'][1]
    assert 'RLE_DICTIONARY' in columns['name'][1]


def test_import_pyarrow_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    with pytest.raises(UsageError, match='pyarrow'):
        convert_parquet.import_pyarrow()


def test_import_pyarrow_too_old(monkeypatch):
    import pyarrow
    monkeypatch.setattr(pyarrow, '__version__', '23.0.0')
    with pytest.raises(UsageError, match='pyarrow>=24'):
        convert_parquet.import_pyarrow()


from nominatim_api.search.query_analyzer_factory import make_query_analyzer  # noqa: E402
from nominatim_api.sql.duckdb_layout import (PARQUET_TABLES_PROPERTY,  # noqa: E402
                                             STORAGE_FORMAT_PROPERTY)


def test_verify_table_detects_missing_rows(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex WHERE place_id > 0',
                                 out, None)
    with pytest.raises(UsageError, match='rows'):
        convert_parquet.verify_table(con, 'placex', 'SELECT * FROM placex', out)


def test_verify_table_detects_type_change(con, tmp_path):
    out = tmp_path / 'placex.parquet'
    con.execute(f"COPY (SELECT * REPLACE (name::VARCHAR AS name) FROM placex) TO '{out}'")
    with pytest.raises(UsageError, match='name'):
        convert_parquet.verify_table(con, 'placex', 'SELECT * FROM placex', out)


def test_verify_table_detects_geometry_crs(con, tmp_path):
    import pyarrow.parquet as pq
    out = tmp_path / 'placex.parquet'
    data = con.execute('SELECT place_id, ST_AsWKB(geometry) AS geometry FROM placex')\
              .to_arrow_table()
    geo = {'version': '1.0.0', 'primary_column': 'geometry',
           'columns': {'geometry': {'encoding': 'WKB', 'geometry_types': []}}}
    pq.write_table(data.replace_schema_metadata({b'geo': json.dumps(geo).encode('utf-8')}),
                   str(out))
    with pytest.raises(UsageError, match='geometry'):
        convert_parquet.verify_table(con, 'placex', 'SELECT place_id, geometry FROM placex', out)


def test_verify_rowids_detects_reordering(con, tmp_path):
    con.execute('CREATE TABLE placex_rowids AS'
                ' SELECT place_id, rowid AS rid FROM placex ORDER BY place_id')
    out = tmp_path / 'placex.parquet'
    convert_parquet.export_table(con, 'placex', 'SELECT * FROM placex ORDER BY place_id DESC',
                                 out, None)
    with pytest.raises(UsageError, match='row order'):
        convert_parquet.verify_rowids(con, out)


@pytest.fixture
def source_db(apiobj):
    apiobj.add_data(
        'properties',
        [{'property': 'tokenizer', 'value': 'icu'},
         {'property': 'tokenizer_import_normalisation', 'value': ':: lower();'},
         {'property': 'tokenizer_import_transliteration', 'value': "'1' > '/1/';"}])
    for i in range(20):
        apiobj.add_placex(place_id=1 + i, osm_type='N', osm_id=i, rank_search=30,
                          name={'name': f'p{i}'}, centroid=(130.0 + i * 0.1, 30.0))
    apiobj.add_search_name(1, names=[10], address=[99], centroid=(130.0, 30.0))

    async def _word():
        async with apiobj.api._async_api.begin() as conn:
            await make_query_analyzer(conn)
            await conn.connection.run_sync(conn.t.meta.tables['word'].create)
    apiobj.async_to_sync(_word())
    return apiobj


def test_convert_writes_parquet_directory(source_db, tmp_path):
    outdir = tmp_path / 'pq'
    source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse', 'search'}))

    names = sorted(p.name for p in outdir.iterdir())
    assert 'placex.parquet' in names and 'nominatim_properties.parquet' in names
    assert all(n.endswith('.parquet') for n in names)  # no leftovers

    props = dict(duckdb.connect().execute(
        f"SELECT property, value FROM '{outdir / 'nominatim_properties.parquet'}'").fetchall())
    assert props[STORAGE_FORMAT_PROPERTY] == 'parquet'
    assert set(props[PARQUET_TABLES_PROPERTY].split(',')) == {n[:-8] for n in names}


def test_convert_uses_table_settings(source_db, tmp_path, monkeypatch):
    monkeypatch.setitem(convert_parquet.PARQUET_TABLE_MAX_ROWS, 'placex', 7)
    outdir = tmp_path / 'pq'
    source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse'}))

    reader = duckdb.connect()
    placex = outdir / 'placex.parquet'
    assert [r[0] for r in reader.execute(
        f"""SELECT max(row_group_num_rows) FROM parquet_metadata('{placex}')
             GROUP BY row_group_id ORDER BY row_group_id""").fetchall()] == [7, 7, 6]
    assert {r[0] for r in reader.execute(
        f"""SELECT DISTINCT path_in_schema FROM parquet_metadata('{placex}')
             WHERE bloom_filter_offset IS NOT NULL""").fetchall()} == {'osm_id', 'place_id'}
    columns = column_encodings(placex)
    assert columns['place_id'][0] == 'LZ4_RAW'
    assert 'RLE_DICTIONARY' not in columns['place_id'][1]
    assert 'RLE_DICTIONARY' in columns['class'][1]


def test_convert_without_search_has_no_search_tables(source_db, tmp_path):
    outdir = tmp_path / 'pq'
    source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse'}))

    assert not (outdir / 'search_name.parquet').exists()
    assert not (outdir / 'word.parquet').exists()


def test_convert_keep_duckdb(source_db, tmp_path):
    outdir = tmp_path / 'pq'
    source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse'},
                                                    keep_duckdb=True))

    dbfile = outdir / convert_parquet.DUCKDB_NAME
    with duckdb.connect(str(dbfile), read_only=True) as kept:
        # the database file itself stays a plain DuckDB database
        assert kept.execute('SELECT count(*) FROM nominatim_properties WHERE property = ?',
                            (STORAGE_FORMAT_PROPERTY, )).fetchone()[0] == 0


def test_convert_refuses_non_empty_directory(source_db, tmp_path):
    outdir = tmp_path / 'pq'
    outdir.mkdir()
    (outdir / 'other.txt').write_text('keep me')

    with pytest.raises(UsageError, match='not empty'):
        source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse'}))
    assert (outdir / 'other.txt').read_text() == 'keep me'


def test_convert_removes_partial_output_on_error(source_db, tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise UsageError('verification failed')
    monkeypatch.setattr(convert_parquet, 'verify_rowids', _fail)
    outdir = tmp_path / 'pq'

    with pytest.raises(UsageError, match='verification failed'):
        source_db.async_to_sync(convert_parquet.convert(None, outdir, {'reverse'}))
    assert not outdir.exists() or not any(outdir.iterdir())
