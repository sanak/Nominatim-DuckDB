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

Places are read from placex by row id where only a few place_ids are
needed. A join on place_id gets only an optional dynamic filter, which
DuckDB uses to skip row groups but not rows. The place_ids are scattered
over the spatially sorted table, so such a join reads the full rows of
hundreds of thousands of places. A join on the row id with up to
`dynamic_or_filter_threshold` (default 50) rows reads only the vectors
that contain the places. The table `placex_rowids` (see
`placex_rowids()`) gives the row id for a place_id and is sorted by
place_id, so that a lookup in it reads only the row groups of the ids.
Join both on place_id and on the row id: with more rows than the
threshold, DuckDB falls back to the place_id filter.

A Parquet export has the same tables with the same rows in the same
order, one file `<table>.parquet` per table. The frontend exposes each
file as a view with the Parquet row number as `rowid`, which therefore
equals the row id of the database file and `placex_rowids.rid`. The
tables in PARQUET_MEMORY_TABLES are loaded into memory instead, with the
Parquet row number in a regular column `rowid` (which takes precedence
over DuckDB's row id pseudo column).
"""
from typing import Dict, Tuple

import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_asyncio

from ..errors import UsageError
from ..typing import SaColumn

# Version of the layout described here. Increase it whenever a change of
# the layout needs a new conversion of existing databases. The converter
# saves it in nominatim_properties under LAYOUT_VERSION_PROPERTY.
LAYOUT_VERSION = 1
LAYOUT_VERSION_PROPERTY = 'duckdb_layout_version'

# Properties of a Parquet export (`nominatim convert --format parquet`):
# the storage format ('parquet'; a database file has no such property)
# and the comma-separated list of exported tables, which includes
# PROPERTIES_TABLE itself.
STORAGE_FORMAT_PROPERTY = 'duckdb_storage_format'
PARQUET_TABLES_PROPERTY = 'duckdb_parquet_tables'
PROPERTIES_TABLE = 'nominatim_properties'

# Tables of a Parquet export that are copied into the shared in-memory
# instance at the first connection instead of being read through a view.
# The Parquet reader has no statistics for the planner and has to map
# place_ids to row ids of search_name with a full scan for every search;
# a native table is about 40 ms faster per search (Japan: 2 GB of RAM).
PARQUET_MEMORY_TABLES: Tuple[str, ...] = ('search_name', )

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


# Table mapping place_id to the row id of the place in placex (`rid`).
PLACEX_ROWID_TABLE = 'placex_rowids'


async def check_layout_version(conn: sa_asyncio.AsyncConnection, location: str) -> None:
    """ Make sure that a database created by `nominatim convert` has the
        layout of this version of the frontend. Databases without the table
        (or view) nominatim_properties are not Nominatim databases and are
        not checked.
    """
    if not await conn.scalar(sa.text("SELECT count(*) FROM information_schema.tables"
                                     " WHERE table_name = 'nominatim_properties'")):
        return

    version = await conn.scalar(sa.text("SELECT max(value) FROM nominatim_properties"
                                        " WHERE property = :name"),
                                {'name': LAYOUT_VERSION_PROPERTY})
    if version != str(LAYOUT_VERSION):
        raise UsageError(f"DuckDB database '{location}' has layout version"
                         f" {version or 'unknown'}, but this version of Nominatim needs"
                         f" layout version {LAYOUT_VERSION}. Create the database again"
                         " with 'nominatim convert'.")


def placex_rowids() -> 'sa.TableClause':
    """ Return the table that maps the place_ids of placex to the row
        ids of the places (columns `place_id` and `rid`).
    """
    return sa.table(PLACEX_ROWID_TABLE, sa.column('place_id', sa.BigInteger),
                    sa.column('rid', sa.BigInteger))


def placex_rowid() -> SaColumn:
    """ Return the row id column of the (unaliased) placex table.
    """
    return sa.literal_column('placex.rowid', sa.BigInteger)


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
