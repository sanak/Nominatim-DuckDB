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
from nominatim_api.sql.sqlalchemy_types import Geometry

duckdb = pytest.importorskip('duckdb')
pytest.importorskip('duckdb_engine')

from nominatim_api.sql.duckdb_async import AsyncDuckDBDialect  # noqa: E402
from nominatim_db.tools import convert_duckdb  # noqa: E402

# Constructs with a SQLite variant whose DuckDB variant is still missing.
# They are only used by the forward search, which is enabled for DuckDB
# in a later step. Remove the entries from this list as they are done.
PENDING_DUCKDB_SUPPORT = {
    'nominatim_api.search.db_search_lookups.LookupAll',
    'nominatim_api.search.db_search_lookups.LookupAny',
    'nominatim_api.search.db_search_lookups.Restrict',
    'nominatim_api.sql.sqlalchemy_functions.RegexpWord',
    'nominatim_api.sql.sqlalchemy_functions.CategoryMatch',
    'nominatim_api.sql.sqlalchemy_types.int_array.ArrayAgg',
    'nominatim_api.sql.sqlalchemy_types.int_array.ArrayContains',
    'nominatim_api.sql.sqlalchemy_types.int_array.ArrayCat',
}

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
    sql = _compile(t.c.linegeo.within_distance(sa.literal_column('g'), sa.text('0.5')))

    assert 'location_property_osmline.linegeo_maxx >= ST_XMin(g) - 0.5' in sql
    assert 'location_property_osmline.linegeo_miny <= ST_YMax(g) + 0.5' in sql
    assert sql.endswith('ST_DWithin(location_property_osmline.linegeo, g, 0.5))')


def test_column_dwithin_without_bbox_columns():
    t = SearchTables(sa.MetaData()).placex
    sql = _compile(t.c.centroid.within_distance(sa.literal_column('g'), sa.text('0.5')))

    assert sql == 'ST_DWithin(placex.centroid, g, 0.5)'


@pytest.fixture
def duckdb_api(apiobj, tmp_path):
    """ A small converted database with helper functions for
        running individual SQL constructs against DuckDB.
    """
    apiobj.add_data('properties', [{'property': 'tokenizer', 'value': 'icu'}])
    apiobj.add_placex(place_id=1, centroid=(139.7671, 35.6812))

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


def _make_synthetic_db(dbfile, nside, step):
    """ Create a database in the layout of `nominatim convert` with
        a grid of nside x nside POIs in placex and all other tables empty.
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
                      osm_type, osm_id, class, type, name, geometry, centroid)
                    SELECT i, 0.0001, 30, 30, 0, 'N', i, 'amenity', 'cafe',
                           '{{"name": "Cafe"}}', pt, pt
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
                                           order=convert_duckdb.KEY_ORDER.get(name))
        con.execute(f'DROP TABLE src_{name}')
    con.execute('CHECKPOINT')
    con.close()


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

    scans = []

    def _find_scans(node, out):
        info = node.get('extra_info') or {}
        # The table name is qualified with the database name.
        if node.get('operator_type') == 'TABLE_SCAN' \
           and info.get('Table', '').split('.')[-1] == 'placex':
            out.append(node['operator_rows_scanned'])
        for child in node.get('children', []):
            _find_scans(child, out)

    api = napi.NominatimAPI(environ={'NOMINATIM_DATABASE_DSN': f'duckdb:dbname={dbfile}'})
    try:
        api._loop.run_until_complete(api._async_api.setup_database())
        engine = api._async_api._engine.sync_engine

        @sa.event.listens_for(engine, 'connect')
        def _enable_profiling(dbapi_con, _):
            cursor = dbapi_con.cursor()
            cursor.execute("PRAGMA enable_profiling = 'json'")
            cursor.execute(f"SET profiling_output = '{profile}'")
            cursor.execute("PRAGMA profiling_mode = 'detailed'")

        @sa.event.listens_for(engine, 'after_cursor_execute')
        def _collect_scans(conn, cursor, statement, *_):
            if 'placex' in statement:
                with open(profile, encoding='utf-8') as fd:
                    _find_scans(json.load(fd), scans)

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
