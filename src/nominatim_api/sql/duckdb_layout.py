# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Physical layout of a DuckDB database created with `nominatim convert`.

The DuckDB database has no indexes. Spatial filters are instead done on
plain bounding box helper columns of type DOUBLE, which allow DuckDB to
skip row groups via their min/max statistics (zonemaps). The tables are
physically sorted so that the row groups are spatially compact.

The bbox helper columns of a geometry column `<col>` are named:

  * `minx`, `miny`, `maxx`, `maxy` when the column is called `geometry`,
  * `<col>_x`, `<col>_y` for point columns (`centroid`). The point is its
    own bounding box, so `bbox_columns()` returns the x/y columns twice,
  * `<col>_minx`, `<col>_miny`, `<col>_maxx`, `<col>_maxy` otherwise.

Use `table_bbox_columns()` to find out which geometry columns of a table
have bbox helper columns.
"""
from typing import Dict, Tuple

# Geometry columns that are stored as points only.
POINT_COLUMNS = ('centroid', )

# Tables which have bbox helper columns, mapped to the geometry column
# that the bbox columns describe. The same column determines the physical
# sort order of the table.
BBOX_TABLES: Dict[str, str] = {
    'placex': 'geometry',
    'placex_place_node_areas': 'geometry',
    'location_property_osmline': 'linegeo',
    'location_property_tiger': 'linegeo',
    'location_postcodes': 'geometry',
    'country_osm_grid': 'geometry',
    'search_name': 'centroid',
}

# Further geometry columns with bbox helper columns, which do not
# influence the sort order. The centroid of placex is filtered by
# distance in POI and near searches.
EXTRA_BBOX_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'placex': ('centroid', ),
}


def table_bbox_columns(table: str) -> Tuple[str, ...]:
    """ Return the geometry columns of the given table which have
        bbox helper columns. The first one determines the sort order.
    """
    if table not in BBOX_TABLES:
        return ()
    return (BBOX_TABLES[table], ) + EXTRA_BBOX_COLUMNS.get(table, ())


def reverse_place_diameter_sql(rank: str) -> str:
    """ Return the SQL for the search radius (in degrees) of a place node
        with the search rank given by the SQL expression `rank`. This is the
        step function `reverse_place_diameter()` of PostgreSQL. The same
        values must be used for the frontend queries and for the extents
        of `placex_place_node_areas`.
    """
    return (f"CAST(CASE WHEN {rank} <= 4 THEN 5.0 WHEN {rank} <= 8 THEN 1.8"
            f" WHEN {rank} <= 12 THEN 0.6 WHEN {rank} <= 17 THEN 0.16"
            f" WHEN {rank} <= 18 THEN 0.08 WHEN {rank} <= 19 THEN 0.04"
            " ELSE 0.02 END AS DOUBLE)")


def bbox_columns(column: str) -> Tuple[str, str, str, str]:
    """ Return the names of the bbox helper columns (minx, miny, maxx, maxy)
        for the given geometry column.
    """
    if column == 'geometry':
        return ('minx', 'miny', 'maxx', 'maxy')
    if column in POINT_COLUMNS:
        return (f'{column}_x', f'{column}_y', f'{column}_x', f'{column}_y')
    return (f'{column}_minx', f'{column}_miny', f'{column}_maxx', f'{column}_maxy')
