# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for the DuckDB variants of the custom SQL constructs.
"""
import json

import pytest
import sqlalchemy as sa

import nominatim_api as napi
from nominatim_api.sql.sqlalchemy_schema import SearchTables
from nominatim_api.sql.sqlalchemy_types import Geometry, IntArray
from nominatim_api.sql.sqlalchemy_functions import CategoryMatch
from nominatim_api.search import db_search_lookups as lookups
from nominatim_api.search.db_search_fields import (FieldRanking, RankedTokens, FieldLookup,
                                                   WeightedStrings, WeightedCategories)
from nominatim_api.search.db_searches import PlaceSearch, NearSearch
from nominatim_api.types import SearchDetails
from nominatim_api.search.query_analyzer_factory import make_query_analyzer

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('duckdb_engine')

from nominatim_api.sql.duckdb_async import AsyncDuckDBDialect  # noqa: E402
from nominatim_db.tools import convert_duckdb  # noqa: E402

# Constructs with a SQLite variant whose DuckDB variant is still missing.
PENDING_DUCKDB_SUPPORT = set()

# The DuckDB row group size. Zonemaps (min/max statistics) are per row group.
ROW_GROUP_SIZE = 122880


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _compile(expr):
    return str(expr.compile(dialect=AsyncDuckDBDialect()))


def test_every_sqlite_override_has_duckdb_override():
    missing = set()
    for base in (sa.sql.ClauseElement, sa.types.TypeEngine):
        for cls in set(_subclasses(base)):
            disp = cls.__dict__.get('_compiler_dispatcher')
            if disp is not None and 'sqlite' in disp.specs and 'duckdb' not in disp.specs:
                missing.add(f'{cls.__module__}.{cls.__name__}')

    assert missing == PENDING_DUCKDB_SUPPORT


def test_column_intersects_uses_bbox_columns():
    t = SearchTables(sa.MetaData()).placex
    other = t.alias('outer')
    sql = _compile(other.c.geometry.intersects(sa.literal_column('g')))

    assert sql == ('("outer".maxx >= ST_XMin(g) AND "outer".minx <= ST_XMax(g)'
                   ' AND "outer".maxy >= ST_YMin(g) AND "outer".miny <= ST_YMax(g))')


def test_column_dwithin_uses_bbox_columns():
    t = SearchTables(sa.MetaData()).osmline
    sql = _compile(t.c.linegeo.within_distance(sa.literal_column('g'), sa.text('0.25 + 0.25')))

    assert 'location_property_osmline.linegeo_maxx >= ST_XMin(g) - (0.25 + 0.25)' in sql
    assert 'location_property_osmline.linegeo_miny <= ST_YMax(g) + (0.25 + 0.25)' in sql
    assert sql.endswith('ST_DWithin(location_property_osmline.linegeo, g, 0.25 + 0.25))')


def test_column_dwithin_without_bbox_columns():
    t = SearchTables(sa.MetaData()).postcode
    sql = _compile(t.c.centroid.within_distance(sa.literal_column('g'), sa.text('0.5')))

    assert sql == 'ST_DWithin(location_postcodes.centroid, g, 0.5)'


def test_column_dwithin_placex_centroid_uses_point_columns():
    t = SearchTables(sa.MetaData()).placex
    sql = _compile(t.c.centroid.within_distance(sa.literal_column('g'), sa.text('0.5')))

    assert sql == ('(placex.centroid_x >= ST_XMin(g) - (0.5)'
                   ' AND placex.centroid_x <= ST_XMax(g) + (0.5)'
                   ' AND placex.centroid_y >= ST_YMin(g) - (0.5)'
                   ' AND placex.centroid_y <= ST_YMax(g) + (0.5)'
                   ' AND ST_DWithin(placex.centroid, g, 0.5))')


def test_column_covered_by_uses_bbox_columns():
    t = SearchTables(sa.MetaData()).placex
    sql = _compile(t.c.centroid.ST_CoveredBy(sa.literal_column('g')))

    assert sql == ('(placex.centroid_x >= ST_XMin(g) AND placex.centroid_x <= ST_XMax(g)'
                   ' AND placex.centroid_y >= ST_YMin(g) AND placex.centroid_y <= ST_YMax(g)'
                   ' AND ST_CoveredBy(placex.centroid, g))')


def test_covered_by_without_bbox_columns():
    t = SearchTables(sa.MetaData()).placex
    sub = sa.select(t.c.centroid, t.c.geometry).subquery('sub')
    sql = _compile(sub.c.centroid.ST_CoveredBy(sub.c.geometry))

    assert sql == 'ST_CoveredBy(sub.centroid, sub.geometry)'


@pytest.fixture
def duckdb_api(apiobj, tmp_path):
    """ A small converted database with helper functions for
        running individual SQL constructs against DuckDB.
    """
    apiobj.add_data('properties', [{'property': 'tokenizer', 'value': 'icu'}])
    apiobj.add_placex(place_id=1, centroid=(139.7671, 35.6812))
    # Place nodes with all the ranks of PG_REVERSE_PLACE_DIAMETER.
    for rank in PG_REVERSE_PLACE_DIAMETER:
        apiobj.add_placex(place_id=100 + rank, osm_type='N', class_='place', type='village',
                          name={'name': 'Village'}, rank_search=rank, rank_address=20,
                          centroid=(139.7671, 35.6812))

    outfile = tmp_path / 'test.duckdb'
    apiobj.async_to_sync(convert_duckdb.convert(None, outfile, {'reverse'}))

    class _Helper:
        api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={outfile}'})

        def scalar(self, sql, params=None):
            async def _run():
                async with self.api._async_api.begin() as conn:
                    return await conn.scalar(sql, params or {})
            return self.api._loop.run_until_complete(_run())

        def scalar_distance(self, pt1, pt2):
            p1 = sa.bindparam('p1', type_=Geometry)
            p2 = sa.bindparam('p2', type_=Geometry)
            return self.scalar(sa.select(p1.distance_spheroid(p2)),
                               {'p1': 'POINT(%f %f)' % pt1, 'p2': 'POINT(%f %f)' % pt2})

    helper = _Helper()
    yield helper
    helper.api.close()


def test_distance_spheroid_tokyo_shinjuku(duckdb_api):
    # Tokyo Sta. (139.7671, 35.6812) -> Shinjuku Sta. (139.7006, 35.6896) ~= 6084 m
    dist = duckdb_api.scalar_distance((139.7671, 35.6812), (139.7006, 35.6896))
    assert dist == pytest.approx(6084, rel=0.01)


def test_distance_spheroid_line_to_point(duckdb_api):
    # Distance to the closest point of the line, here the point itself.
    p1 = sa.bindparam('p1', type_=Geometry)
    p2 = sa.bindparam('p2', type_=Geometry)
    dist = duckdb_api.scalar(sa.select(p1.distance_spheroid(p2)),
                             {'p1': 'LINESTRING(139.7006 35.6896, 139.7671 35.6812)',
                              'p2': 'POINT(139.7671 35.6812)'})
    assert dist == pytest.approx(0.0, abs=0.001)


@pytest.mark.parametrize('wkt,exp',
                         [('POINT(1 2.5)', 'POINT(1 2.5)'),
                          ('LINESTRING(0 0, 1 1)', 'LINESTRING(0 0,1 1)'),
                          ('MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))',
                           'MULTIPOLYGON(((0 0,1 0,1 1,0 0)),((5 5,6 5,6 6,5 5)))')])
def test_as_text_like_postgis(duckdb_api, wkt, exp):
    geom = sa.bindparam('g', type_=Geometry)
    assert duckdb_api.scalar(sa.select(sa.func.ST_AsText(geom)), {'g': wkt}) == exp


def test_as_geojson(duckdb_api):
    geom = sa.bindparam('g', type_=Geometry)
    out = duckdb_api.scalar(sa.select(sa.func.ST_AsGeoJSON(geom, 7)),
                            {'g': 'LINESTRING(1 2.5, 10.05 20)'})
    assert out == '{"type":"LineString","coordinates":[[1,2.5],[10.05,20]]}'


def test_as_svg_like_postgis(duckdb_api):
    geom = sa.bindparam('g', type_=Geometry)
    out = duckdb_api.scalar(sa.select(sa.func.ST_AsSVG(geom, 0, 7)),
                            {'g': 'POLYGON((23 34, 23.1 34, 23.1 34.1, 23 34))'})
    assert out == 'M 23 -34 L 23.1 -34 23.1 -34.1 Z'


def test_as_kml_unsupported(duckdb_api):
    geom = sa.bindparam('g', type_=Geometry)
    with pytest.raises(napi.UsageError, match='KML'):
        duckdb_api.scalar(sa.select(sa.func.ST_AsKML(geom, 7)), {'g': 'POINT(1 2)'})


def test_geometry_result_is_ewkb(duckdb_api):
    t = SearchTables(sa.MetaData()).placex
    out = duckdb_api.scalar(sa.select(t.c.centroid).where(t.c.place_id == 1))
    assert napi.Point.from_wkb(out) == napi.Point(139.7671, 35.6812)


# Values of the SQL function reverse_place_diameter() of PostgreSQL.
PG_REVERSE_PLACE_DIAMETER = {4: 5.0, 8: 1.8, 12: 0.6, 17: 0.16, 18: 0.08,
                             19: 0.04, 20: 0.02, 30: 0.02}


@pytest.mark.parametrize('rank,diameter', PG_REVERSE_PLACE_DIAMETER.items())
def test_is_below_reverse_distance_like_postgres(duckdb_api, rank, diameter):
    def _is_below(dist):
        return duckdb_api.scalar(sa.select(sa.func.IsBelowReverseDistance(
                                     sa.literal(dist), sa.literal(rank))))

    assert _is_below(diameter * 0.99)
    assert not _is_below(diameter * 1.01)


@pytest.mark.parametrize('rank,diameter', PG_REVERSE_PLACE_DIAMETER.items())
def test_intersects_reverse_distance_like_postgres(duckdb_api, rank, diameter):
    # Also checks the extent of the place in placex_place_node_areas.
    t = SearchTables(sa.MetaData()).placex

    def _intersects(dist):
        pt = sa.bindparam('pt', type_=Geometry)
        sql = sa.select(sa.func.count()).where(t.c.place_id == 100 + rank)\
                .where(sa.func.IntersectsReverseDistance(t, pt))
        return duckdb_api.scalar(sql, {'pt': f'POINT({139.7671 + dist} 35.6812)'}) == 1

    assert _intersects(diameter * 0.99)
    assert not _intersects(diameter * 1.01)


@pytest.mark.parametrize('rank,diameter', PG_REVERSE_PLACE_DIAMETER.items())
def test_place_node_areas_like_postgres(rank, diameter):
    con = duckdb.connect()
    con.load_extension('spatial')
    con.execute("""CREATE TABLE placex AS
                   SELECT 1 AS place_id, ST_Point(10, 20) AS geometry,
                          ?::SMALLINT AS rank_search, 20::SMALLINT AS rank_address,
                          'N' AS osm_type, NULL::BIGINT AS linked_place_id""", [rank])
    extent = con.execute('SELECT ST_XMin(geometry), ST_XMax(geometry),'
                         ' ST_YMin(geometry), ST_YMax(geometry)'
                         ' FROM (' + convert_duckdb.NODE_AREAS_SQL + ')').fetchone()

    assert extent == pytest.approx((10 - diameter, 10 + diameter,
                                    20 - diameter, 20 + diameter))


def _make_synthetic_db(dbfile, nside, step, place_id='i'):
    """ Create a database in the layout of `nominatim convert` with
        a grid of nside x nside POIs in placex and all other tables empty.
        The POI number i is at (130 + (i % nside) * step, 30 + (i // nside) * step)
        and gets the place_id computed by the SQL expression `place_id`.
    """
    con = duckdb.connect(str(dbfile))
    con.load_extension('spatial')
    con.execute('SET preserve_insertion_order = true')
    for table in SearchTables(sa.MetaData()).meta.sorted_tables:
        coldefs = ', '.join(f'"{c.name}" {convert_duckdb._duckdb_type(c.type)}'
                            for c in table.c)
        con.execute(f'CREATE TABLE src_{table.name} ({coldefs})')
    con.execute(f"""INSERT INTO src_placex
                     (place_id, importance, rank_address, rank_search, indexed_status,
                      osm_type, osm_id, class, type, name, geometry, centroid, categories)
                    SELECT {place_id}, 0.0001, 30, 30, 0, 'N', i, 'amenity', 'cafe',
                           '{{"name": "Cafe"}}', pt, pt, ['osm.amenity.cafe']
                      FROM (SELECT i, ST_Point(130 + (i % {nside}) * {step},
                                               30 + (i // {nside}) * {step}) AS pt
                              FROM range({nside * nside}) t(i))
                     ORDER BY random()""")
    con.execute('CREATE TABLE src_placex_place_node_areas AS '
                + convert_duckdb.NODE_AREAS_SQL.replace('FROM placex', 'FROM src_placex'))
    tables = [t.name for t in SearchTables(sa.MetaData()).meta.sorted_tables]
    for name in tables + ['placex_place_node_areas']:
        convert_duckdb.create_sorted_table(con, name, f'src_{name}',
                                           geom_column=convert_duckdb.BBOX_TABLES.get(name),
                                           order=convert_duckdb.KEY_ORDER.get(name),
                                           extra_columns=convert_duckdb.EXTRA_BBOX_COLUMNS
                                                                       .get(name, ()))
        con.execute(f'DROP TABLE src_{name}')
    convert_duckdb.create_placex_rowids(con)
    con.execute('CHECKPOINT')
    con.close()


class _ScanProfiler:
    """ Collects the rows scanned by every table scan of the given table
        for all statements run through the API, using the JSON profiling
        output of DuckDB.
    """

    def __init__(self, api, profile, table, statement_filter=''):
        self.scans = []
        self.nodes = []
        engine = api._async_api._engine.sync_engine

        def _find_scans(node):
            info = node.get('extra_info') or {}
            # The table name is qualified with the database name.
            if node.get('operator_type') == 'TABLE_SCAN' \
               and info.get('Table', '').split('.')[-1] == table:
                self.scans.append(node['operator_rows_scanned'])
                self.nodes.append(node)
            for child in node.get('children', []):
                _find_scans(child)

        @sa.event.listens_for(engine, 'connect')
        def _enable_profiling(dbapi_con, _):
            cursor = dbapi_con.cursor()
            cursor.execute("PRAGMA enable_profiling = 'json'")
            cursor.execute(f"SET profiling_output = '{profile}'")
            cursor.execute("PRAGMA profiling_mode = 'detailed'")

        @sa.event.listens_for(engine, 'after_cursor_execute')
        def _collect_scans(conn, cursor, statement, *_):
            if table in statement and statement_filter in statement:
                with open(profile, encoding='utf-8') as fd:
                    _find_scans(json.load(fd))


def test_reverse_prunes_placex_row_groups(tmp_path):
    """ A reverse query must only scan the few row groups of placex
        that cover the area around the query point.
    """
    nside, step = 1600, 0.01
    dbfile = tmp_path / 'grid.duckdb'
    _make_synthetic_db(dbfile, nside, step)
    profile = tmp_path / 'profile.json'
    nrows = nside * nside
    assert nrows >= 20 * ROW_GROUP_SIZE

    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    try:
        api._loop.run_until_complete(api._async_api.setup_database())
        profiler = _ScanProfiler(api, profile, 'placex')
        scans = profiler.scans

        assert api.reverse((135.5, 35.5)).place_id == 550 * nside + 550
        assert api.reverse((131.031, 38.709)).place_id == 871 * nside + 103
        # Only does an area search on placex, which finds nothing.
        assert api.reverse((135.505, 35.505), zoom=10) is None
    finally:
        api.close()

    print('rows scanned in placex:', scans, 'of', nrows)
    assert len(scans) >= 3
    for rows in scans:
        assert rows < 0.2 * nrows


def test_address_details_prune_place_addressline(tmp_path):
    """ The address lines of a place and of its parent must be looked
        up with a join that skips the row groups of place_addressline
        that do not contain the places (the table is sorted by place_id).
    """
    nside, nlines = 100, 3000000
    dbfile = tmp_path / 'address.duckdb'
    _make_synthetic_db(dbfile, nside, 0.01)
    con = duckdb.connect(str(dbfile))
    con.execute('SET preserve_insertion_order = true')
    con.execute('DROP TABLE place_addressline')
    # 10 address lines for each place, pointing to the places in placex.
    con.execute(f"""CREATE TABLE src AS
                    SELECT (i // 10)::BIGINT AS place_id,
                           ((i * 7) % {nside * nside})::BIGINT AS address_place_id,
                           0.0 AS distance, true AS fromarea, (i % 2 = 0) AS isaddress
                      FROM range({nlines}) t(i) ORDER BY random()""")
    convert_duckdb.create_sorted_table(con, 'place_addressline', 'src',
                                       order=convert_duckdb.KEY_ORDER['place_addressline'])
    con.execute('DROP TABLE src')
    # Place 50 gets its address from its parent with a far-away place_id.
    con.execute('UPDATE placex SET parent_place_id = 250000 WHERE place_id = 50')
    con.execute('CHECKPOINT')
    con.close()

    profile = tmp_path / 'profile.json'
    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    try:
        api._loop.run_until_complete(api._async_api.setup_database())
        profiler = _ScanProfiler(api, profile, 'place_addressline')
        results = api.lookup([napi.PlaceID(50), napi.PlaceID(7000)], address_details=True)
    finally:
        api.close()

    def _lines(*pids):
        return sorted(((pid * 10 + i) * 7) % (nside * nside) for pid in pids for i in range(10))

    assert [r.place_id for r in results] == [50, 7000]
    for res, pids in zip(results, ((50, 250000), (7000, ))):
        assert sorted(a.place_id for a in res.address_rows
                      if a.place_id is not None and a.place_id != res.place_id) == _lines(*pids)

    print('rows scanned in place_addressline:', profiler.scans, 'of', nlines)
    assert len(profiler.scans) == 1
    assert profiler.scans[0] <= 3 * ROW_GROUP_SIZE


###########################################################################
# Forward search

@pytest.fixture
def duckdb_search(apiobj, tmp_path):
    """ A converted database with search tables for running the
        search constructs against DuckDB.
    """
    apiobj.add_data('properties',
                    [{'property': 'tokenizer', 'value': 'icu'},
                     {'property': 'tokenizer_import_normalisation', 'value': ':: lower();'},
                     {'property': 'tokenizer_import_transliteration', 'value': "'1' > '/1/';"}])
    apiobj.add_placex(place_id=1, class_='amenity', type='restaurant', housenumber='12a')
    apiobj.add_placex(place_id=2, class_='amenity', type='fast_food', housenumber='3')
    apiobj.add_placex(place_id=3, class_='tourism', type='hotel', housenumber='112')
    apiobj.add_placex(place_id=4, class_='amenity', type='cafe', categories=[])
    apiobj.add_search_name(1, names=[1, 2, 3], address=[10, 11])
    apiobj.add_search_name(2, names=[2, 3], address=[11])
    apiobj.add_search_name(3, names=[4], address=[10])

    async def _word():
        async with apiobj.api._async_api.begin() as conn:
            await make_query_analyzer(conn)
            await conn.connection.run_sync(conn.t.meta.tables['word'].create)
    apiobj.async_to_sync(_word())

    outfile = tmp_path / 'search.duckdb'
    apiobj.async_to_sync(convert_duckdb.convert(None, outfile, {'search'}))

    class _Helper:
        api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={outfile}'})
        t = SearchTables(sa.MetaData())

        def rows(self, sql):
            async def _run():
                async with self.api._async_api.begin() as conn:
                    return (await conn.execute(sql)).all()
            return self.api._loop.run_until_complete(_run())

        def place_ids(self, table, where):
            return sorted(r[0] for r in self.rows(sa.select(table.c.place_id).where(where)))

    helper = _Helper()
    yield helper
    helper.api.close()


@pytest.mark.parametrize('lookup,column,tokens,expected',
                         [(lookups.LookupAll, 'name_vector', [2, 3], [1, 2]),
                          (lookups.LookupAll, 'name_vector', [1, 2], [1]),
                          (lookups.LookupAll, 'nameaddress_vector', [10], [1, 3]),
                          (lookups.LookupAll, 'name_vector', [2, 99], []),
                          (lookups.LookupAll, 'name_vector', [3, 3], [1, 2]),
                          (lookups.LookupAll, 'name_vector', [], []),
                          (lookups.LookupAll, 'name_vector', [99], []),
                          (lookups.LookupAny, 'name_vector', [1, 4], [1, 3]),
                          (lookups.LookupAny, 'nameaddress_vector', [11, 99], [1, 2]),
                          (lookups.LookupAny, 'name_vector', [99], []),
                          (lookups.Restrict, 'name_vector', [2, 3], [1, 2]),
                          (lookups.Restrict, 'nameaddress_vector', [10, 11], [1])])
def test_search_name_lookups(duckdb_search, lookup, column, tokens, expected):
    t = duckdb_search.t.search_name
    assert duckdb_search.place_ids(t, lookup(t, column, tokens)) == expected


def test_weigh_search(duckdb_search):
    t = duckdb_search.t.search_name
    ranking = FieldRanking('name_vector', 1.0,
                           [RankedTokens(0.1, [1, 2]), RankedTokens(0.5, [3])])
    rows = duckdb_search.rows(sa.select(t.c.place_id, ranking.sql_penalty(t)))

    assert sorted(rows) == [(1, 0.1), (2, 0.5), (3, 1.0)]


def test_array_cat_and_contains(duckdb_search):
    t = duckdb_search.t.search_name
    both = t.c.name_vector + t.c.nameaddress_vector

    assert duckdb_search.place_ids(t, both.contains(sa.type_coerce([3, 10], IntArray))) == [1]
    assert duckdb_search.place_ids(t, both.contains(sa.type_coerce([2, 11], IntArray))) == [1, 2]


def test_array_agg(duckdb_search):
    t = duckdb_search.t.search_name
    out = duckdb_search.rows(sa.select(sa.func.ArrayAgg(t.c.place_id)))[0][0]
    assert sorted(out) == [1, 2, 3]


def test_regexp_word(duckdb_search):
    t = duckdb_search.t.placex
    assert duckdb_search.place_ids(t, sa.func.RegexpWord('12A|3', t.c.housenumber)) == [1, 2]


@pytest.mark.parametrize('text,words,expected',
                         [('1番地', '1', False), ('東京駅1', '1', False),
                          ('丁目3', '3', False), ('ä1', '1', False),
                          ('１', '１', True), ('1番地', '1番地', True),
                          ('Ä1', 'ä1', True), ('1-3', '3', True),
                          ('1 2', '2', True), ('1_2', '1', False)])
def test_regexp_word_unicode_boundaries(duckdb_search, text, words, expected):
    sql = sa.select(sa.func.RegexpWord(words, sa.literal(text)))
    assert duckdb_search.rows(sql)[0][0] == expected


@pytest.mark.parametrize('category,expected',
                         [('osm.amenity', [1, 2]),
                          ('osm.amenity.restaurant', [1]),
                          ('osm.amen', []),
                          ('osm.amenity.rest', [])])
def test_category_match(duckdb_search, category, expected):
    t = duckdb_search.t.placex
    assert duckdb_search.place_ids(t, CategoryMatch(t, category)) == expected


def test_category_match_empty_list_is_false(duckdb_search):
    t = duckdb_search.t.placex
    assert 4 in duckdb_search.place_ids(t, sa.not_(CategoryMatch(t, 'osm.amenity')))


def test_word_lookup_prunes_reverse_search_name(tmp_path):
    """ Word lookups in reverse_search_name must only read the row
        groups that contain the words, which works because the table
        is sorted by word.
    """
    nplaces = 1500000
    dbfile = tmp_path / 'words.duckdb'
    _make_synthetic_db(dbfile, 1, 0.01)
    con = duckdb.connect(str(dbfile))
    con.load_extension('spatial')
    con.execute('SET preserve_insertion_order = true')
    con.execute('DROP TABLE search_name')
    # Every place has its own name and address word, in random order.
    con.execute(f"""CREATE TABLE search_name AS
                    SELECT i::BIGINT AS place_id, 0.0001 AS importance,
                           30::SMALLINT AS search_rank, 30::SMALLINT AS address_rank,
                           [i]::INTEGER[] AS name_vector,
                           [i + {nplaces}]::INTEGER[] AS nameaddress_vector,
                           'jp' AS country_code, ST_Point(135, 35) AS centroid,
                           135.0 AS centroid_x, 35.0 AS centroid_y
                      FROM range({nplaces}) t(i) ORDER BY random()""")
    con.execute('CREATE TABLE src_reverse_search_name AS ' + convert_duckdb.REVERSE_SEARCH_SQL)
    convert_duckdb.create_sorted_table(con, 'reverse_search_name', 'src_reverse_search_name',
                                       order=convert_duckdb.KEY_ORDER['reverse_search_name'])
    con.execute('DROP TABLE src_reverse_search_name')
    con.execute('CHECKPOINT')
    nrows = con.execute('SELECT count(*) FROM reverse_search_name').fetchone()[0]
    con.close()
    assert nrows >= 20 * ROW_GROUP_SIZE

    profile = tmp_path / 'profile.json'
    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    try:
        api._loop.run_until_complete(api._async_api.setup_database())
        profiler = _ScanProfiler(api, profile, 'reverse_search_name')
        t = SearchTables(sa.MetaData()).search_name

        async def _lookup(where):
            async with api._async_api.begin() as conn:
                return sorted(r[0] for r in await conn.execute(sa.select(t.c.place_id)
                                                               .where(where)))

        def lookup(where):
            return api._loop.run_until_complete(_lookup(where))

        assert lookup(lookups.LookupAll(t, 'name_vector', [1234])) == [1234]
        assert lookup(lookups.LookupAll(t, 'nameaddress_vector',
                                        [nplaces + 77])) == [77]
        assert lookup(lookups.LookupAny(t, 'name_vector', [5, 1400000])) == [5, 1400000]
    finally:
        api.close()

    print('rows scanned in reverse_search_name:', profiler.scans, 'of', nrows)
    assert len(profiler.scans) == 3
    for rows in profiler.scans:
        assert rows <= 2 * ROW_GROUP_SIZE


def test_search_name_lookup_reads_only_candidates(tmp_path):
    """ A search for a rare word must only read the vectors of the rows
        of search_name that contain the word. All other row groups
        are skipped through the row ids of the candidates.
    """
    nplaces = 1000000
    dbfile = tmp_path / 'search.duckdb'
    _make_synthetic_db(dbfile, 1, 0.01)
    con = duckdb.connect(str(dbfile))
    con.load_extension('spatial')
    con.execute('SET preserve_insertion_order = true')
    con.execute('DROP TABLE search_name')
    # Spatially sorted like the converter does it, so that the place_ids
    # of the candidates are spread over all row groups. The rare words
    # 1 to 8 are in the names of 8 places each and word 100 in the
    # address of every second place. The place_ids must be unique and
    # are scattered (1000003 is a prime).
    con.execute(f"""CREATE TABLE src AS
                    SELECT (i * 7919) % 1000003 AS place_id, 0.0001 * (i % 97) AS importance,
                           30::SMALLINT AS search_rank, 30::SMALLINT AS address_rank,
                           ([CASE WHEN i % 125000 < 8 THEN i % 125000 + 1 ELSE 1000 + i END]
                            || [(2000000 + (i * k) % 70000)::INTEGER for k in range(1, 9)]
                           )::INTEGER[] AS name_vector,
                           ([CASE WHEN i % 2 = 0 THEN 100 ELSE 101 END]
                            || [(3000000 + (i // 100 * 79 + k * 1047) % 20000)::INTEGER
                                for k in range(1, 50)])::INTEGER[] AS nameaddress_vector,
                           'jp' AS country_code,
                           ST_Point(130 + (i * 7919 % 1000) * 0.01,
                                    30 + (i * 104729 % 1000) * 0.01) AS centroid
                      FROM range({nplaces}) t(i)""")
    convert_duckdb.create_sorted_table(con, 'search_name', 'src',
                                       geom_column=convert_duckdb.BBOX_TABLES['search_name'])
    con.execute('DROP TABLE src')
    assert con.execute('SELECT count(DISTINCT place_id) = count(*)'
                       ' FROM search_name').fetchone()[0]
    con.execute('CREATE TABLE src AS ' + convert_duckdb.REVERSE_SEARCH_SQL
                .replace('GROUP BY', 'WHERE word < 200 GROUP BY'))
    convert_duckdb.create_sorted_table(con, 'reverse_search_name', 'src',
                                       order=convert_duckdb.KEY_ORDER['reverse_search_name'])
    con.execute('DROP TABLE src')
    con.execute('CHECKPOINT')
    con.close()

    profile = tmp_path / 'profile.json'
    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    try:
        api._loop.run_until_complete(api._async_api.setup_database())
        profiler = _ScanProfiler(api, profile, 'search_name')
        t = SearchTables(sa.MetaData()).search_name
        ranking = FieldRanking('name_vector', 1.0, [RankedTokens(0.1, [3])])

        async def _search(*where):
            sql = sa.select(t.c.place_id, ranking.sql_penalty(t).label('penalty'))\
                    .where(*where).order_by(t.c.importance.desc()).limit(1000)
            async with api._async_api.begin() as conn:
                return sorted(await conn.execute(sql))

        def search(*where):
            return api._loop.run_until_complete(_search(*where))

        rows = search(lookups.LookupAll(t, 'name_vector', [3]))
        expected = sorted((i * 7919) % 1000003 for i in range(2, nplaces, 125000))
        assert [r[0] for r in rows] == expected
        assert all(r[1] == 0.1 for r in rows)
        rows = search(lookups.LookupAny(t, 'name_vector', [4, 5]),
                      lookups.Restrict(t, 'nameaddress_vector', [100]))
        assert len(rows) == 8
    finally:
        api.close()

    print('search_name scans (rows scanned, rows read, filters):',
          [(n['operator_rows_scanned'], n['operator_cardinality'],
            n['extra_info'].get('Filters'), n['extra_info'].get('Dynamic Filters', '')[:40])
           for n in profiler.nodes])
    vector_scans = [n for n in profiler.nodes if 'name_vector' in n['extra_info']['Projections']]
    assert len(vector_scans) == 2
    for node in vector_scans:
        # Only the vectors that contain a candidate are read.
        assert 'rowid IN' in node['extra_info'].get('Dynamic Filters', '')
        assert node['operator_cardinality'] < 0.01 * nplaces


###########################################################################
# Places read by place_id

GRID_SIDE = 1600
# The place_ids of the grid are scattered over the spatially sorted placex
# like on real data (2560021 is a prime).
GRID_PRIME = 2560021


def _grid_id(i):
    return (i * 7919) % GRID_PRIME


@pytest.fixture(scope='module')
def grid_db(tmp_path_factory):
    """ A grid of GRID_SIDE x GRID_SIDE cafes (step 0.01 degrees) in the
        layout of `nominatim convert`, place_ids from `_grid_id()`. The
        places 50 and 2000050 have ten address places each. About every
        1000th place is in search_name, the word 7 in the names of the
        places 1000 and 2001000.
    """
    nplaces = GRID_SIDE * GRID_SIDE
    dbfile = tmp_path_factory.mktemp('grid') / 'grid.duckdb'
    _make_synthetic_db(dbfile, GRID_SIDE, 0.01, place_id=f'(i * 7919) % {GRID_PRIME}')
    con = duckdb.connect(str(dbfile))
    con.load_extension('spatial')
    con.execute('SET preserve_insertion_order = true')
    con.execute('DROP TABLE place_addressline')
    con.execute(f"""CREATE TABLE src AS
                    SELECT ((p::BIGINT * 7919) % {GRID_PRIME})::BIGINT AS place_id,
                           (((p::BIGINT + 1 + k * 250007) % {nplaces}) * 7919
                            % {GRID_PRIME})::BIGINT AS address_place_id,
                           0.0 AS distance, true AS fromarea, true AS isaddress
                      FROM (VALUES (50), (2000050)) v(p), range(10) r(k)""")
    convert_duckdb.create_sorted_table(con, 'place_addressline', 'src',
                                       order=convert_duckdb.KEY_ORDER['place_addressline'])
    con.execute('DROP TABLE src')
    con.execute('DROP TABLE search_name')
    con.execute(f"""CREATE TABLE src AS
                    SELECT place_id, importance, rank_search AS search_rank,
                           rank_address AS address_rank,
                           [CASE WHEN place_id IN ({_grid_id(1000)}, {_grid_id(2001000)}) THEN 7
                                 ELSE 100 + place_id END]::INTEGER[] AS name_vector,
                           [99]::INTEGER[] AS nameaddress_vector,
                           'jp' AS country_code, centroid
                      FROM placex
                     WHERE place_id % 1000 = 0
                           OR place_id IN ({_grid_id(1000)}, {_grid_id(2001000)})""")
    convert_duckdb.create_sorted_table(con, 'search_name', 'src',
                                       geom_column=convert_duckdb.BBOX_TABLES['search_name'])
    con.execute('DROP TABLE src')
    con.execute('CREATE TABLE src AS ' + convert_duckdb.REVERSE_SEARCH_SQL)
    convert_duckdb.create_sorted_table(con, 'reverse_search_name', 'src',
                                       order=convert_duckdb.KEY_ORDER['reverse_search_name'])
    con.execute('DROP TABLE src')
    con.execute('CHECKPOINT')
    con.close()

    return dbfile, nplaces


@pytest.fixture
def grid_api(grid_db):
    dbfile, _ = grid_db
    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    api._loop.run_until_complete(api._async_api.setup_database())
    yield api
    api.close()


def _run_search(api, search, details):
    async def _run():
        async with api._async_api.begin() as conn:
            return await search(conn, details)
    return api._loop.run_until_complete(_run())


# Number of rows in a DuckDB vector.
VECTOR_SIZE = 2048


def _assert_few_placex_rows(profiler, nrows, nplaces):
    """ Every scan of placex must either skip most row groups or select
        the rows of the `nplaces` places by row id, which reads at most
        one vector per place. (A filter on place_id is only used to skip
        row groups and, depending on the data, whole row groups are read.)
    """
    print('placex scans (rows scanned, rows emitted, dynamic filters):',
          [(n['operator_rows_scanned'], n['operator_cardinality'],
            str(n['extra_info'].get('Dynamic Filters', ''))[:80]) for n in profiler.nodes])
    assert profiler.nodes
    for node in profiler.nodes:
        if node['operator_rows_scanned'] >= 0.2 * nrows:
            assert 'rowid' in str(node['extra_info'].get('Dynamic Filters', ''))
            assert node['operator_cardinality'] <= nplaces * VECTOR_SIZE


def test_place_search_reads_placex_by_rowid(grid_db, grid_api, tmp_path):
    """ The places found in search_name must be read from placex through
        their row ids, not by scanning the row groups their place_ids
        fall into.
    """
    class _Data:
        penalty = 0.0
        postcodes = WeightedStrings([], [])
        countries = WeightedStrings([], [])
        qualifiers = WeightedCategories([], [])
        lookups = [FieldLookup('name_vector', [7], lookups.LookupAll)]
        rankings = []
        housenumbers = None

    profiler = _ScanProfiler(grid_api, tmp_path / 'profile.json', 'placex', 'searches')
    results = _run_search(grid_api, PlaceSearch(0.0, _Data(), 2, False).lookup,
                          SearchDetails())

    assert sorted(r.place_id for r in results) == sorted([_grid_id(1000), _grid_id(2001000)])
    _assert_few_placex_rows(profiler, grid_db[1], 2)


def test_address_details_read_placex_by_rowid(grid_db, grid_api, tmp_path):
    """ The address places of a result must be read from placex through
        their row ids.
    """
    nplaces = grid_db[1]
    profiler = _ScanProfiler(grid_api, tmp_path / 'profile.json', 'placex',
                             'place_addressline')
    results = grid_api.lookup([napi.PlaceID(_grid_id(50)), napi.PlaceID(_grid_id(2000050))],
                              address_details=True)

    assert [r.place_id for r in results] == [_grid_id(50), _grid_id(2000050)]
    for res, i in zip(results, (50, 2000050)):
        assert sorted(a.place_id for a in res.address_rows if a.place_id != res.place_id) \
            == sorted(_grid_id((i + 1 + k * 250007) % nplaces) for k in range(10))
    _assert_few_placex_rows(profiler, nplaces, 20)


def test_near_search_reads_only_places_near_the_anchor(grid_db, grid_api, tmp_path):
    """ A category search around a place must only read the places
        of the category in the search area of the place and then
        the full rows of the results.
    """
    anchor = 800 * GRID_SIDE + 800
    search = NearSearch(0.1, WeightedCategories([('amenity', 'cafe')], [0.0]), None)
    profiler = _ScanProfiler(grid_api, tmp_path / 'profile.json', 'placex')

    async def _lookup(conn, details):
        results = napi.SearchResults()
        await search.lookup_category(results, conn, [_grid_id(anchor)], ('amenity', 'cafe'),
                                     0.0, details)
        return results

    results = _run_search(grid_api, _lookup, SearchDetails(max_results=5))

    assert results[0].place_id == _grid_id(anchor)
    assert {r.place_id for r in results[1:]} == {_grid_id(anchor + d) for d in
                                                 (-1, 1, -GRID_SIDE, GRID_SIDE)}
    _assert_few_placex_rows(profiler, grid_db[1], 5)
