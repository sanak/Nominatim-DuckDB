# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for exporting a database to DuckDB.
"""
import pytest

from nominatim_api.search.query_analyzer_factory import make_query_analyzer

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('duckdb_engine')

from nominatim_api.sql.duckdb_layout import (BBOX_TABLES, bbox_columns,  # noqa: E402
                                             table_bbox_columns, LAYOUT_VERSION,
                                             LAYOUT_VERSION_PROPERTY)
from nominatim_db.tools import convert_duckdb  # noqa: E402
from nominatim_db.errors import UsageError  # noqa: E402

# DuckDB row group size: zonemaps (min/max) are kept per row group.
ROW_GROUP_SIZE = 122880


def test_bbox_columns_naming():
    assert bbox_columns('geometry') == ('minx', 'miny', 'maxx', 'maxy')
    assert bbox_columns('linegeo') == ('linegeo_minx', 'linegeo_miny',
                                       'linegeo_maxx', 'linegeo_maxy')
    assert bbox_columns('centroid') == ('centroid_x', 'centroid_y',
                                        'centroid_x', 'centroid_y')


def test_table_bbox_columns():
    assert table_bbox_columns('placex') == ('geometry', 'centroid')
    assert table_bbox_columns('search_name') == ('centroid', )
    assert table_bbox_columns('location_property_osmline') == ('linegeo', )
    assert table_bbox_columns('word') == ()


@pytest.fixture
def converted(apiobj, tmp_path):
    apiobj.add_data(
        'properties',
        [{'property': 'tokenizer', 'value': 'icu'},
         {'property': 'tokenizer_import_normalisation', 'value': ':: lower();'},
         {'property': 'tokenizer_import_transliteration', 'value': "'1' > '/1/';"}])
    for i in range(10):
        for j in range(10):
            apiobj.add_placex(place_id=1 + i * 10 + j, osm_type='N', osm_id=i * 10 + j,
                              rank_search=20, rank_address=20,
                              name={'name': f'p{i}-{j}', 'name:ja': '東京'},
                              centroid=(130.0 + i * 0.5, 30.0 + j * 0.5))
    apiobj.add_placex(place_id=500, osm_type='R', rank_search=8, rank_address=8,
                      centroid=(132.0, 32.0),
                      geometry='POLYGON((130 30, 135 30, 135 35, 130 35, 130 30))')
    apiobj.add_osmline(place_id=600, geometry='LINESTRING(131 31, 131.01 31.01)')
    apiobj.add_postcode(place_id=700, postcode='100-0001', country_code='jp')
    apiobj.add_country('jp', 'POLYGON((122 20, 154 20, 154 46, 122 46, 122 20))')
    for pid, words in ((1, [30, 10]), (2, [20, 10]), (3, [40]), (4, [5, 30]), (5, [7, 7])):
        apiobj.add_search_name(pid, names=words, address=[99, words[0]],
                               centroid=(130.0 + pid, 30.0))

    async def _word():
        async with apiobj.api._async_api.begin() as conn:
            await make_query_analyzer(conn)
            await conn.connection.run_sync(conn.t.meta.tables['word'].create)
    apiobj.async_to_sync(_word())
    apiobj.add_word_table([(10, 'tokyo', 'W', 'tokyo', None),
                           (5, 'abc', 'W', 'abc', {'count': 3})])

    outfile = tmp_path / 'out.duckdb'
    apiobj.async_to_sync(convert_duckdb.convert(None, outfile, {'reverse', 'search', 'details'}))

    con = duckdb.connect(str(outfile), read_only=True)
    con.load_extension('spatial')
    yield outfile, con
    con.close()


def test_convert_single_file(converted, tmp_path):
    outfile, _ = converted
    assert [p.name for p in tmp_path.iterdir()] == [outfile.name]


def test_convert_creates_no_indexes(converted):
    _, con = converted
    assert con.execute('SELECT count(*) FROM duckdb_indexes()').fetchone()[0] == 0


def test_convert_copies_tables(converted):
    _, con = converted
    tables = {r[0] for r in con.execute('SELECT table_name FROM duckdb_tables()').fetchall()}
    assert {'placex', 'placex_place_node_areas', 'placex_rowids', 'search_name', 'word',
            'reverse_search_name', 'location_property_osmline'} <= tables
    assert con.execute('SELECT count(*) FROM placex').fetchone()[0] == 101
    assert con.execute('SELECT count(*) FROM placex_place_node_areas').fetchone()[0] == 100
    assert con.execute("SELECT name->>'name:ja' FROM placex WHERE place_id = 1")\
              .fetchone()[0] == '東京'
    assert con.execute("SELECT name_vector FROM search_name WHERE place_id = 1")\
              .fetchone()[0] == [30, 10]
    assert con.execute("SELECT info->>'count' FROM word WHERE word_id = 5")\
              .fetchone()[0] == '3'
    assert con.execute("SELECT categories FROM placex WHERE place_id = 1")\
              .fetchone()[0] == ['osm.highway.residential']
    assert con.execute("SELECT ST_AsText(geometry) FROM placex WHERE place_id = 1")\
              .fetchone()[0] == 'POINT (130 30)'


@pytest.mark.parametrize('table,geom', [(t, g) for t in sorted(BBOX_TABLES)
                                        for g in table_bbox_columns(t)])
def test_convert_bbox_columns(converted, table, geom):
    _, con = converted
    minx, miny, maxx, maxy = bbox_columns(geom)
    bad = con.execute(f"""SELECT count(*) FROM {table}
                          WHERE {geom} IS NOT NULL
                                AND ({minx} IS NULL OR {minx} != ST_XMin({geom})
                                     OR {miny} IS NULL OR {miny} != ST_YMin({geom})
                                     OR {maxx} IS NULL OR {maxx} != ST_XMax({geom})
                                     OR {maxy} IS NULL OR {maxy} != ST_YMax({geom}))
                       """).fetchone()[0]
    assert bad == 0


def test_convert_reverse_search_name_one_row_per_word_and_place(converted):
    _, con = converted
    rows = con.execute('SELECT "column", word, place_id FROM reverse_search_name'
                       ' ORDER BY rowid').fetchall()
    vectors = con.execute('SELECT place_id, name_vector, nameaddress_vector'
                          ' FROM search_name').fetchall()
    expected = {(col, word, pid) for pid, names, address in vectors
                for col, words in (('name_vector', names), ('nameaddress_vector', address))
                for word in words}

    assert rows == sorted(expected)
    assert ('name_vector', 10, 1) in rows
    assert rows.count(('name_vector', 7, 5)) == 1


def test_convert_sets_layout_version(converted):
    _, con = converted
    assert con.execute('SELECT value FROM nominatim_properties WHERE property = ?',
                       (LAYOUT_VERSION_PROPERTY, )).fetchall() == [(str(LAYOUT_VERSION), )]


def test_convert_placex_rowids(converted):
    _, con = converted
    rows = con.execute('SELECT place_id, rid FROM placex_rowids ORDER BY rowid').fetchall()

    assert rows == sorted(con.execute('SELECT place_id, rowid FROM placex').fetchall())


def test_spatial_sort_prunes_row_groups(tmp_path):
    """ A grid of 1.2M points mixed with large polygons must be sorted such
        that the bbox min/max of most row groups exclude a point query.
    """
    con = duckdb.connect(str(tmp_path / 'sort.duckdb'))
    con.load_extension('spatial')
    con.execute("""CREATE TABLE src AS SELECT * FROM (
                     SELECT ST_Point(130 + (i % 1100) * 0.01, 25 + (i // 1100) * 0.01) AS geometry
                       FROM range(1200000) t(i)
                     UNION ALL
                     SELECT ST_Buffer(ST_Point(130 + (i % 40) * 0.3, 25 + (i // 40) * 0.3), 1.5)
                       FROM range(2000) t(i))
                     ORDER BY random()""")
    convert_duckdb.create_sorted_table(con, 'placex', 'src', geom_column='geometry')

    stats = con.execute(f"""SELECT min(minx), min(miny), max(maxx), max(maxy)
                              FROM placex GROUP BY rowid // {ROW_GROUP_SIZE}""").fetchall()
    con.close()
    assert len(stats) >= 10

    for x in (131.0, 133.0, 135.5, 138.0, 140.0):
        for y in (26.0, 28.0, 30.5, 33.0, 35.0):
            pruned = sum(1 for (x0, y0, x1, y1) in stats
                         if x < x0 or x > x1 or y < y0 or y > y1)
            assert pruned / len(stats) >= 0.5, (x, y)


def test_connect_disables_extension_autoloading(tmp_path):
    con = convert_duckdb._connect(tmp_path / 'out.duckdb', '')
    try:
        settings = dict(con.execute(
            """SELECT name, value FROM duckdb_settings()
                WHERE name IN ('autoinstall_known_extensions',
                               'autoload_known_extensions')""").fetchall())
    finally:
        con.close()
    assert settings == {'autoinstall_known_extensions': 'false',
                        'autoload_known_extensions': 'false'}


def test_connect_missing_spatial_extension(tmp_path):
    extdir = tmp_path / 'ext'
    extdir.mkdir()
    outfile = tmp_path / 'out.duckdb'

    with pytest.raises(UsageError, match='NOMINATIM_DUCKDB_EXTENSION_DIR'):
        convert_duckdb._connect(outfile, str(extdir))

    assert list(extdir.iterdir()) == []  # nothing was downloaded
    assert not outfile.exists()


@pytest.mark.parametrize('ids,ok', [([1, 2, 3], True), ([1, 2, 2], False)])
def test_check_unique_place_ids(ids, ok):
    con = duckdb.connect()
    con.execute('CREATE TABLE search_name AS SELECT unnest(?::BIGINT[]) AS place_id', [ids])
    if ok:
        convert_duckdb.check_unique_place_ids(con)
    else:
        with pytest.raises(UsageError, match='1 duplicate place_ids'):
            convert_duckdb.check_unique_place_ids(con)
