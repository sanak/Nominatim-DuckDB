# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
DuckDB variants of the custom SQL constructs. (EXPERIMENTAL)

All DuckDB-specific compilation lives in this module. Constructs
that compile to the same SQL as on PostgreSQL get an explicit DuckDB
variant nonetheless, so that it is visible that they have been checked.

A DuckDB database has no spatial indexes. Spatial filters on table
columns are instead done on the bbox helper columns described in
`duckdb_layout`, which allow DuckDB to skip row groups through their
min/max statistics.

Output of geometries in KML format is not supported because DuckDB
has no function for it. Requesting it raises a UsageError.
"""
from typing import Any, List, Optional

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles

from ..errors import UsageError
from .duckdb_layout import BBOX_TABLES, bbox_columns
from .sqlalchemy_types.geometry import (Geometry, Geometry_DistanceSpheroid,
                                        Geometry_IsLineLike, Geometry_IsAreaLike,
                                        Geometry_IntersectsBbox,
                                        Geometry_ColumnIntersectsBbox,
                                        Geometry_ColumnDWithin,
                                        FUNCTION_ALIAS_CLASSES)
from .sqlalchemy_types.key_value import KeyValueConcat
from .sqlalchemy_types.ltree import CategoryArray, _LtreeArrayCast
from .sqlalchemy_functions import (PlacexGeometryReverseLookuppolygon,
                                   IntersectsReverseDistance, IsBelowReverseDistance,
                                   IsAddressPoint, CrosscheckNames, JsonArrayEach,
                                   Greatest)

DIALECT = 'duckdb'


def _bbox_column_sql(col: Any, compiler: 'sa.Compiled', **kw: Any) -> Optional[List[str]]:
    """ Return the SQL for the bbox helper columns (minx, miny, maxx, maxy)
        of the given geometry column or None, if the table has no bbox
        columns for it. The columns are qualified in the same way as the
        geometry column, so that aliases are respected.
    """
    table = getattr(col, 'table', None)
    if isinstance(table, sa.sql.expression.Alias):
        table = table.element
    if not isinstance(table, sa.Table) or BBOX_TABLES.get(table.name) != col.name:
        return None

    colsql = compiler.process(col, **kw)
    prefix = colsql[:colsql.rfind('.') + 1]
    return [prefix + c for c in bbox_columns(col.name)]


def _bbox_overlap_sql(bbox: List[str], geom: str, dist: str = '') -> str:
    """ SQL for checking that the bbox columns overlap with the bbox of
        the geometry SQL `geom`, optionally expanded by `dist`.
        The comparisons with a constant are pushed down into the table
        scan, where they skip row groups.
    """
    minx, miny, maxx, maxy = bbox
    sub = f' - ({dist})' if dist else ''
    add = f' + ({dist})' if dist else ''
    return (f"{maxx} >= ST_XMin({geom}){sub} AND {minx} <= ST_XMax({geom}){add}"
            f" AND {maxy} >= ST_YMin({geom}){sub} AND {miny} <= ST_YMax({geom}){add}")


###########################################################################
# Geometry type and functions

@compiles(Geometry, DIALECT)
def _duckdb_geometry_col_spec(*args: Any, **kwargs: Any) -> str:
    return 'GEOMETRY'


@compiles(Geometry_DistanceSpheroid, DIALECT)
def _duckdb_distance_spheroid(element: Geometry_DistanceSpheroid,
                              compiler: 'sa.Compiled', **kw: Any) -> str:
    # ST_Distance_Spheroid() only works on points in lat/lon axis order.
    # Use the closest points of the two geometries, like PostGIS does.
    geom1, geom2 = (compiler.process(c, **kw) for c in element.clauses)
    return ("ST_Distance_Spheroid("
            f"ST_FlipCoordinates(ST_ClosestPoint({geom1}, {geom2}))::POINT_2D, "
            f"ST_FlipCoordinates(ST_ClosestPoint({geom2}, {geom1}))::POINT_2D)")


@compiles(Geometry_IsLineLike, DIALECT)
def _duckdb_is_line_like(element: Geometry_IsLineLike,
                         compiler: 'sa.Compiled', **kw: Any) -> str:
    return "ST_GeometryType(%s) IN ('LINESTRING', 'MULTILINESTRING')" % \
               compiler.process(element.clauses, **kw)


@compiles(Geometry_IsAreaLike, DIALECT)
def _duckdb_is_area_like(element: Geometry_IsAreaLike,
                         compiler: 'sa.Compiled', **kw: Any) -> str:
    return "ST_GeometryType(%s) IN ('POLYGON', 'MULTIPOLYGON')" % \
               compiler.process(element.clauses, **kw)


@compiles(Geometry_IntersectsBbox, DIALECT)
def _duckdb_intersects(element: Geometry_IntersectsBbox,
                       compiler: 'sa.Compiled', **kw: Any) -> str:
    arg1, arg2 = (compiler.process(c, **kw) for c in element.clauses)
    return f"ST_Intersects_Extent({arg1}, {arg2})"


@compiles(Geometry_ColumnIntersectsBbox, DIALECT)
def _duckdb_intersects_column(element: Geometry_ColumnIntersectsBbox,
                              compiler: 'sa.Compiled', **kw: Any) -> str:
    # Same semantics as '&&' in PostGIS: the bounding boxes intersect.
    col, geom = list(element.clauses)
    geomsql = compiler.process(geom, **kw)
    bbox = _bbox_column_sql(col, compiler, **kw)
    if bbox is None:
        return f"ST_Intersects_Extent({compiler.process(col, **kw)}, {geomsql})"

    return f"({_bbox_overlap_sql(bbox, geomsql)})"


@compiles(Geometry_ColumnDWithin, DIALECT)
def _duckdb_dwithin_column(element: Geometry_ColumnDWithin,
                           compiler: 'sa.Compiled', **kw: Any) -> str:
    col, geom, dist = list(element.clauses)
    colsql, geomsql, distsql = (compiler.process(c, **kw) for c in (col, geom, dist))
    exact = f"ST_DWithin({colsql}, {geomsql}, {distsql})"
    bbox = _bbox_column_sql(col, compiler, **kw)
    if bbox is None:
        return exact

    return f"({_bbox_overlap_sql(bbox, geomsql, distsql)} AND {exact})"


def _register_function(name: str, rettype: Any, template: str, nargs: int = 99) -> None:
    """ Compile the given function on DuckDB using the template, where
        '{0}', '{1}' etc. are replaced with the arguments. Arguments
        after the first `nargs` ones are dropped. Functions without a
        SQLite alias get a function class of their own, which compiles
        to the plain function call on all other dialects.
    """
    func_class = FUNCTION_ALIAS_CLASSES.get(name)
    if func_class is None:
        func_class = type(name, (sa.sql.functions.GenericFunction, ), {
            "type": rettype(),
            "name": name,
            "identifier": name,
            "inherit_cache": True})

    def _duckdb_impl(element: Any, compiler: Any, **kw: Any) -> str:
        # Dropped arguments must not be processed, or their bind
        # parameters would still be sent to the database.
        args = list(element.clauses)[:nargs]
        return template.format(*(compiler.process(c, **kw) for c in args))

    compiles(func_class, DIALECT)(_duckdb_impl)


def _duckdb_unsupported(element: Any, compiler: Any, **kw: Any) -> str:
    raise UsageError(f"Function {element.name} is not supported with a DuckDB database.")


# EWKB as hex string with the SRID added, as PostGIS returns it.
_register_function('ST_AsEWKB', sa.Text,
                   r"regexp_replace(ST_AsHEXWKB({0}), '^(.{{8}})00', '\120E6100000')")
_register_function('ST_GeomFromEWKT', Geometry,
                   r"ST_GeomFromText(nullif(regexp_replace({0}, '^SRID=[0-9]+;', ''), ''))")
# DuckDB has no precision parameter and writes all digits. Integral
# coordinates are written without '.0' like PostGIS does.
_register_function('ST_AsGeoJSON', sa.Text,
                   r"regexp_replace(ST_AsGeoJSON({0}), '([0-9])\.0([],])', '\1\2', 'g')",
                   nargs=1)
_register_function('ST_LineLocatePoint', sa.Float, "ST_LineLocatePoint({0}, {1})")
_register_function('ST_LineInterpolatePoint', Geometry, "ST_LineInterpolatePoint({0}, {1})")
# The WKT of DuckDB has blanks after commas and before brackets.
_register_function('ST_AsText', sa.Text,
                   r"regexp_replace(ST_AsText({0}), '\s*([(),])\s*', '\1', 'g')")
# Geometries are always in WGS84, the SRID parameter is dropped.
_register_function('ST_GeomFromText', Geometry, "ST_GeomFromText({0})", nargs=1)
# ST_Collect is a scalar function on a list of geometries in DuckDB.
_register_function('ST_Collect', Geometry, "ST_Collect(list({0}))")
compiles(FUNCTION_ALIAS_CLASSES['ST_AsKML'], DIALECT)(_duckdb_unsupported)


@compiles(FUNCTION_ALIAS_CLASSES['ST_AsSVG'], DIALECT)
def _duckdb_as_svg(element: Any, compiler: 'sa.Compiled', **kw: Any) -> str:
    geom, rel, prec = (compiler.process(c, **kw) for c in element.clauses)
    return f"ST_AsSVG({geom}, CAST({rel} AS BOOLEAN), {prec})"


###########################################################################
# Custom types

@compiles(KeyValueConcat, DIALECT)
def _duckdb_json_concat(element: KeyValueConcat, compiler: 'sa.Compiled', **kw: Any) -> str:
    arg1, arg2 = (compiler.process(c, **kw) for c in element.clauses)
    return f"json_merge_patch({arg1}, coalesce({arg2}, '{{}}'))"


@compiles(CategoryArray, DIALECT)
def _duckdb_category_col_spec(*args: Any, **kwargs: Any) -> str:
    return 'VARCHAR[]'


@compiles(_LtreeArrayCast, DIALECT)
def _duckdb_ltree_array_cast(element: _LtreeArrayCast,
                             compiler: 'sa.Compiled', **kw: Any) -> str:
    return compiler.process(element.clauses, **kw)


###########################################################################
# Functions for reverse geocoding and lookup

@compiles(PlacexGeometryReverseLookuppolygon, DIALECT)
def _duckdb_reverse_lookup_polygon(element: PlacexGeometryReverseLookuppolygon,
                                   compiler: 'sa.Compiled', **kw: Any) -> str:
    return ("(ST_GeometryType(placex.geometry) in ('POLYGON', 'MULTIPOLYGON')"
            " AND placex.rank_address between 4 and 25"
            " AND placex.name is not null"
            " AND placex.indexed_status = 0"
            " AND placex.linked_place_id is null)")


@compiles(IntersectsReverseDistance, DIALECT)
def _duckdb_reverse_place_diameter(element: IntersectsReverseDistance,
                                   compiler: 'sa.Compiled', **kw: Any) -> str:
    geom1, rank, geom2 = (compiler.process(c, **kw) for c in element.clauses)
    table = element.tablename
    areas = [f'placex_place_node_areas.{c}' for c in bbox_columns('geometry')]

    # placex_place_node_areas contains the expanded bbox of the places,
    # so that the bbox check of the area is exact.
    return (f"({table}.rank_address between 4 and 25"
            f" AND {table}.name is not null"
            f" AND {table}.linked_place_id is null"
            f" AND {table}.osm_type = 'N'"
            f" AND ST_Intersects_Extent({geom1},"
            f" ST_Expand({geom2}, 14.0 * exp(-0.2 * {rank}) - 0.03))"
            f" AND {table}.place_id IN"
            f" (SELECT place_id FROM placex_place_node_areas"
            f" WHERE {_bbox_overlap_sql(areas, geom2)}))")


@compiles(IsBelowReverseDistance, DIALECT)
def _duckdb_is_below_reverse_distance(element: IsBelowReverseDistance,
                                      compiler: 'sa.Compiled', **kw: Any) -> str:
    dist, rank = (compiler.process(c, **kw) for c in element.clauses)
    return f"{dist} < 14.0 * exp(-0.2 * {rank}) - 0.03"


@compiles(IsAddressPoint, DIALECT)
def _duckdb_is_address_point(element: IsAddressPoint,
                             compiler: 'sa.Compiled', **kw: Any) -> str:
    rank, hnr, name, address = (compiler.process(c, **kw) for c in element.clauses)
    return (f"({rank} = 30 AND json_extract({address}, '$._inherited') IS NULL"
            f" AND ({hnr} IS NOT NULL"
            f" OR json_extract({name}, '$.\"addr:housename\"') IS NOT NULL))")


@compiles(CrosscheckNames, DIALECT)
def _duckdb_crosscheck_names(element: CrosscheckNames,
                             compiler: 'sa.Compiled', **kw: Any) -> str:
    arg1, arg2 = (compiler.process(c, **kw) for c in element.clauses)
    return (f"coalesce(list_has_any(json_extract_string({arg1}, '$.*'),"
            f" json_extract_string({arg2}, '$[*]')), false)")


@compiles(JsonArrayEach, DIALECT)
def _duckdb_json_array_each(element: JsonArrayEach,
                            compiler: 'sa.Compiled', **kw: Any) -> str:
    return "json_each(%s)" % compiler.process(element.clauses, **kw)


@compiles(Greatest, DIALECT)
def _duckdb_greatest(element: Greatest, compiler: 'sa.Compiled', **kw: Any) -> str:
    return "greatest(%s)" % compiler.process(element.clauses, **kw)
