# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Exporting a Nominatim database to DuckDB. (EXPERIMENTAL)

The resulting database is a single file, which can be used read-only
with the Python frontend (DSN `duckdb:dbname=<file>`).

The database has no indexes at all, to keep the file small. Lookups rely
on DuckDB skipping row groups through their min/max statistics instead:

  * Tables filtered spatially get bbox helper columns (DOUBLE) for their
    geometry column and are physically sorted so that each row group covers
    a compact area. See `nominatim_api.sql.duckdb_layout` for the tables
    and the naming of the columns: `minx/miny/maxx/maxy` for a column
    called `geometry`, `<col>_x/<col>_y` for points (`centroid`) and
    `<col>_minx/..._maxy` otherwise. Use `bbox_columns()` to get the names.
    Some tables have bbox columns for a second geometry column
    (`EXTRA_BBOX_COLUMNS`, e.g. the centroid of placex).
  * The spatial sort order puts geometries into size tiers first (so that
    a few huge polygons do not widen the bbox of every row group) and
    then orders by the Hilbert value of the bbox centre, using the extent
    of the table as bounds.
  * Tables accessed by key are sorted by that key (`word` by word_token,
    `reverse_search_name` by column and word, `place_addressline` by
    place_id).
  * `reverse_search_name` has one row per word and place (in long format,
    not a list of places per word): DuckDB decodes the lists of a whole
    vector of rows to read one of them, and the vectors of the frequent
    words hold hundreds of millions of place ids on a country.
  * `placex_rowids` maps each place_id to the row id of the place in
    placex and is sorted by place_id. The frontend reads places by row id
    through it, see `nominatim_api.sql.duckdb_layout`.
  * The search relies on place_id being unique in `search_name` (as the
    unique index in PostgreSQL guarantees), which is checked after copying.

Data is converted as follows: geometries via WKB, hstore and JSON columns
to JSON, integer arrays to INTEGER[] and category (ltree) arrays to VARCHAR[].

The environment variables DUCKDB_MEMORY_LIMIT and DUCKDB_TEMP_DIR
optionally set the memory limit and spill directory of DuckDB during
the conversion.
"""
from typing import Set, Any, Dict, Optional, Union, TextIO, Sequence
import datetime as dt
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import duckdb
import sqlalchemy as sa

import nominatim_api as napi
from nominatim_api.search.query_analyzer_factory import make_query_analyzer
from nominatim_api.sql.duckdb_layout import (BBOX_TABLES, EXTRA_BBOX_COLUMNS,
                                             POINT_COLUMNS, PLACEX_ROWID_TABLE,
                                             LAYOUT_VERSION, LAYOUT_VERSION_PROPERTY,
                                             bbox_columns, reverse_place_diameter_sql)
from nominatim_api.sql.sqlalchemy_types import (Geometry, IntArray, KeyValueStore,
                                                CategoryArray, Json)

from ..errors import UsageError

LOG = logging.getLogger()

# Physical sort order for tables that are looked up by key.
KEY_ORDER = {
    'place_addressline': 'place_id',
    'placex_entrance': 'place_id',
    'word': 'word_token',
    'reverse_search_name': '"column", word, place_id',
}

# Size tiers (largest bbox side in degrees) for the spatial sort order.
SIZE_TIERS = (0.01, 0.2)

# Number of rows collected in an intermediate file before loading them.
CHUNK_SIZE = 100000

# Maximum size of a row in the intermediate files. Large polygons in
# hex-encoded WKB easily exceed the default of DuckDB (16MB).
MAX_OBJECT_SIZE = 2**30

# The places extended by their search radius in reverse geocoding.
NODE_AREAS_SQL = f"""
    SELECT place_id, ST_Expand(geometry, {reverse_place_diameter_sql('rank_search')}) AS geometry
      FROM placex
     WHERE rank_address BETWEEN 5 AND 25
           AND osm_type = 'N' AND linked_place_id IS NULL"""

# One row per word and place. The vectors may contain a word twice.
REVERSE_SEARCH_SQL = """
    SELECT 'name_vector' AS "column", unnest(list_distinct(name_vector)) AS word, place_id
      FROM search_name
    UNION ALL
    SELECT 'nameaddress_vector', unnest(list_distinct(nameaddress_vector)), place_id
      FROM search_name"""


async def convert(project_dir: Optional[Union[str, Path]],
                  outfile: Path, options: Set[str]) -> None:
    """ Export an existing database to DuckDB. The resulting database
        will be usable against the Python frontend of Nominatim.
    """
    outfile = Path(outfile)
    api = napi.NominatimAPIAsync(project_dir)

    try:
        dest = _connect(outfile, api.config.DUCKDB_EXTENSION_DIR)
        try:
            async with api.begin() as src:
                writer = DuckDBWriter(src, dest, outfile, options)
                await writer.write()
            dest.execute('CHECKPOINT')
        finally:
            dest.close()
    finally:
        await api.close()


def _connect(outfile: Path, extension_dir: str) -> 'duckdb.DuckDBPyConnection':
    """ Open the output database for writing and load the spatial extension.
    """
    # Same extension policy as the frontend: never download extensions.
    config: Dict[str, Any] = {'autoinstall_known_extensions': False,
                              'autoload_known_extensions': False}
    if extension_dir:
        config['extension_directory'] = extension_dir
    con = duckdb.connect(str(outfile), config=config)

    try:
        con.execute('LOAD spatial')
    except duckdb.Error as err:
        con.close()
        for suffix in ('', '.wal'):
            Path(str(outfile) + suffix).unlink(missing_ok=True)
        where = f"directory '{extension_dir}'" if extension_dir else 'default directory'
        raise UsageError(
            f"Cannot load the DuckDB spatial extension from the {where}: {err}\n"
            "Install it beforehand, for example with: python3 -c \"import duckdb; "
            "duckdb.connect().install_extension('spatial')\" and set "
            "NOMINATIM_DUCKDB_EXTENSION_DIR to the extension directory if it is "
            "not the DuckDB default.") from err

    if os.environ.get('DUCKDB_MEMORY_LIMIT'):
        con.execute(f"SET memory_limit = {_quote(os.environ['DUCKDB_MEMORY_LIMIT'])}")
    if os.environ.get('DUCKDB_TEMP_DIR'):
        con.execute(f"SET temp_directory = {_quote(os.environ['DUCKDB_TEMP_DIR'])}")
    # The sorted tables are created with CREATE TABLE AS ... ORDER BY.
    con.execute('SET preserve_insertion_order = true')

    return con


def _quote(value: Union[str, Path]) -> str:
    """ Quote a string literal for DuckDB.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_type(coltype: Any) -> str:
    """ Return the DuckDB type for the given SQLAlchemy column type.
    """
    if isinstance(coltype, Geometry):
        return 'GEOMETRY'
    if isinstance(coltype, (KeyValueStore, Json)):
        return 'JSON'
    if isinstance(coltype, IntArray):
        return 'INTEGER[]'
    if isinstance(coltype, CategoryArray):
        return 'VARCHAR[]'
    if isinstance(coltype, sa.BigInteger):
        return 'BIGINT'
    if isinstance(coltype, sa.SmallInteger):
        return 'SMALLINT'
    if isinstance(coltype, sa.Integer):
        return 'INTEGER'
    if isinstance(coltype, sa.Float):
        return 'DOUBLE'
    if isinstance(coltype, sa.Boolean):
        return 'BOOLEAN'
    if isinstance(coltype, sa.DateTime):
        return 'TIMESTAMPTZ' if coltype.timezone else 'TIMESTAMP'
    if isinstance(coltype, sa.String):
        return 'VARCHAR'
    raise RuntimeError(f"Column type {coltype!r} not supported for DuckDB export.")


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value)}.")


def create_sorted_table(con: 'duckdb.DuckDBPyConnection', name: str, source: str,
                        geom_column: Optional[str] = None,
                        order: Optional[str] = None,
                        extra_columns: Sequence[str] = ()) -> None:
    """ Create the table `name` from the content of table `source`.

        With a `geom_column`, the bbox helper columns for that column are
        added and the rows are sorted spatially. Otherwise the rows are
        sorted by the SQL expression `order`, if given. The geometry columns
        in `extra_columns` get bbox helper columns, too, but do not
        influence the sort order.
    """
    bbox_sql = [_bbox_sql(col) for col in extra_columns]

    if geom_column is not None:
        geom = f'"{geom_column}"'
        minx, miny, maxx, maxy = bbox_columns(geom_column)
        bbox_sql.insert(0, _bbox_sql(geom_column))
        if geom_column in POINT_COLUMNS:
            xmid, ymid = minx, miny
            size = None
        else:
            xmid, ymid = f'(({minx} + {maxx}) / 2)', f'(({miny} + {maxy}) / 2)'
            size = f'greatest({maxx} - {minx}, {maxy} - {miny})'

        order = _spatial_order(con, source, geom, xmid, ymid, size)

    sql = f'SELECT *{"".join(", " + b for b in bbox_sql)} FROM {source}'

    if order is not None:
        sql = f'SELECT * FROM ({sql}) ORDER BY {order}'

    con.execute(f'CREATE TABLE {name} AS {sql}')


def _bbox_sql(column: str) -> str:
    """ Return the select expressions for the bbox helper columns
        of the given geometry column.
    """
    geom = f'"{column}"'
    minx, miny, maxx, maxy = bbox_columns(column)
    if column in POINT_COLUMNS:
        return f'ST_X({geom}) AS {minx}, ST_Y({geom}) AS {miny}'

    return (f'ST_XMin({geom}) AS {minx}, ST_YMin({geom}) AS {miny},'
            f' ST_XMax({geom}) AS {maxx}, ST_YMax({geom}) AS {maxy}')


def _spatial_order(con: 'duckdb.DuckDBPyConnection', source: str, geom: str,
                   xmid: str, ymid: str, size: Optional[str]) -> Optional[str]:
    """ Return the ORDER BY expression that sorts the table by size tier and
        then along a Hilbert curve over the extent of the table.
    """
    extent = con.execute(f"""SELECT min(ST_XMin({geom})), min(ST_YMin({geom})),
                                    max(ST_XMax({geom})), max(ST_YMax({geom}))
                               FROM {source}""").fetchone()
    if extent is None or extent[0] is None:
        return None

    xmin, ymin, xmax, ymax = (float(v) for v in extent)
    # Avoid degenerate bounds for the Hilbert curve.
    xmax = max(xmax, xmin + 1e-6)
    ymax = max(ymax, ymin + 1e-6)

    hilbert = f"""ST_Hilbert({xmid}, {ymid},
                             ST_Extent(ST_MakeEnvelope({xmin!r}, {ymin!r},
                                                       {xmax!r}, {ymax!r})))"""
    if size is None:
        return hilbert

    tiers = ' '.join(f'WHEN {size} < {limit!r} THEN {i}' for i, limit in enumerate(SIZE_TIERS))
    return f'(CASE {tiers} ELSE {len(SIZE_TIERS)} END), {hilbert}'


class DuckDBWriter:
    """ Worker class which creates a new DuckDB database.

        Data is first copied into a staging database next to the output
        file, from where the final, sorted tables are created. This keeps
        the output file free of the space of temporary tables.
    """

    def __init__(self, src: napi.SearchConnection,
                 dest: 'duckdb.DuckDBPyConnection', outfile: Path,
                 options: Set[str]) -> None:
        self.src = src
        self.dest = dest
        self.outfile = outfile
        self.options = options
        self.stage = outfile.with_name(outfile.name + '.stage')

    async def write(self) -> None:
        """ Create the database structure and copy the data from
            the source database to the destination.
        """
        self._remove_stage()  # leftovers of an aborted run
        tmpdir = tempfile.mkdtemp(prefix='nominatim-convert-', dir=self.outfile.parent)
        self.dest.execute(f'ATTACH {_quote(self.stage)} AS stage')
        try:
            for table in self.src.t.meta.sorted_tables:
                if table.name == 'search_name' and 'search' not in self.options:
                    continue
                if table.name != 'word':
                    await self.copy_table(table, Path(tmpdir))

            if 'search' in self.options:
                await make_query_analyzer(self.src)
                await self.copy_table(self.src.t.meta.tables['word'], Path(tmpdir))

            self.create_derived_table('placex_place_node_areas', NODE_AREAS_SQL)
            if 'search' in self.options:
                LOG.warning('Creating reverse search table')
                self.create_derived_table('reverse_search_name', REVERSE_SEARCH_SQL)

            set_layout_version(self.dest)
        finally:
            self.dest.execute('DETACH stage')
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._remove_stage()

    def _remove_stage(self) -> None:
        for suffix in ('', '.wal'):
            Path(str(self.stage) + suffix).unlink(missing_ok=True)

    async def copy_table(self, table: sa.Table, tmpdir: Path) -> None:
        """ Copy the content of the given table into the staging database
            and then create the final table from it.
        """
        LOG.warning("Copying '%s'", table.name)
        coldefs = ', '.join(f'"{c.name}" {_duckdb_type(c.type)}' for c in table.c)
        self.dest.execute(f'CREATE TABLE stage."{table.name}" ({coldefs})')

        is_geom = [isinstance(c.type, Geometry) for c in table.c]
        sql = sa.select(*(sa.func.encode(sa.func.ST_AsBinary(c), 'hex').label(c.name)
                          if g else c.label(c.name)
                          for c, g in zip(table.c, is_geom)))
        jsontypes = ', '.join(f"{_quote(c.name)}: "
                              f"{_quote('VARCHAR' if g else _duckdb_type(c.type))}"
                              for c, g in zip(table.c, is_geom))
        colexprs = ', '.join(f'ST_GeomFromHEXWKB("{c.name}")' if g else f'"{c.name}"'
                             for c, g in zip(table.c, is_geom))
        chunkfile = tmpdir / f'{table.name}.ndjson'
        load_sql = f"""INSERT INTO stage."{table.name}"
                       SELECT {colexprs}
                         FROM read_json({_quote(chunkfile)}, format = 'newline_delimited',
                                        maximum_object_size = {MAX_OBJECT_SIZE},
                                        columns = {{{jsontypes}}})"""

        names = [c.name for c in table.c]
        fd: Optional[TextIO] = None
        nrows = 0
        try:
            async_result = await self.src.connection.stream(sql)
            async for partition in async_result.partitions(10000):
                if fd is None:
                    fd = open(chunkfile, 'w', encoding='utf-8')
                for row in partition:
                    json.dump(dict(zip(names, row)), fd,
                              ensure_ascii=False, default=_json_default)
                    fd.write('\n')
                nrows += len(partition)
                if nrows >= CHUNK_SIZE:
                    fd.close()
                    fd = None
                    self.dest.execute(load_sql)
                    nrows = 0
            if fd is not None:
                fd.close()
                fd = None
                if nrows > 0:
                    self.dest.execute(load_sql)
        finally:
            if fd is not None:
                fd.close()
            chunkfile.unlink(missing_ok=True)

        self.finish_table(table.name)

    def create_derived_table(self, name: str, query: str) -> None:
        """ Compute a table from the already copied data via staging.
        """
        self.dest.execute(f'CREATE TABLE stage."{name}" AS {query}')
        self.finish_table(name)

    def finish_table(self, name: str) -> None:
        """ Move the table from the staging database into the output
            database, adding bbox columns and sorting as necessary.
        """
        create_sorted_table(self.dest, f'"{name}"', f'stage."{name}"',
                            geom_column=BBOX_TABLES.get(name),
                            order=KEY_ORDER.get(name),
                            extra_columns=EXTRA_BBOX_COLUMNS.get(name, ()))
        self.dest.execute(f'DROP TABLE stage."{name}"')
        if name == 'search_name':
            check_unique_place_ids(self.dest)
        elif name == 'placex':
            create_placex_rowids(self.dest)


def set_layout_version(con: 'duckdb.DuckDBPyConnection') -> None:
    """ Save the version of the layout in nominatim_properties, so that
        the frontend can refuse files of another layout.
    """
    con.execute('DELETE FROM nominatim_properties WHERE property = ?',
                (LAYOUT_VERSION_PROPERTY, ))
    con.execute('INSERT INTO nominatim_properties (property, value) VALUES (?, ?)',
                (LAYOUT_VERSION_PROPERTY, str(LAYOUT_VERSION)))


def create_placex_rowids(con: 'duckdb.DuckDBPyConnection') -> None:
    """ Create the table that maps the place_ids of the final placex table
        to the row ids of the places, sorted by place_id. placex must
        not be changed afterwards.
    """
    con.execute(f'CREATE TABLE {PLACEX_ROWID_TABLE} AS'
                ' SELECT place_id, rowid AS rid FROM placex ORDER BY place_id')


def check_unique_place_ids(con: 'duckdb.DuckDBPyConnection') -> None:
    """ Make sure that place_id is unique in search_name. The search
        selects the rows of search_name by place_id via reverse_search_name
        and does not check the search vectors again. PostgreSQL guarantees
        this through the unique index idx_search_name_place_id.
    """
    row = con.execute('SELECT count(*) - count(DISTINCT place_id) FROM search_name').fetchone()
    if row is not None and row[0] != 0:
        raise UsageError(f"Table search_name of the source database has {row[0]}"
                         " duplicate place_ids. Cannot convert.")
