# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Exporting a Nominatim database to Parquet files. (EXPERIMENTAL)

The export is made from a DuckDB database created by `convert_duckdb`
and has the same tables with the same rows in the same physical order,
one Parquet file per table. The frontend reads the files through views
(DSN `duckdb:parquet=<directory or URL>`).

The files are written with pyarrow, because the row groups need to be
cut by size: DuckDB's own writer has a minimum of 2048 rows per row
group. See `PARQUET_ROWGROUP_BYTES` in `nominatim_api.sql.duckdb_layout`.
Geometries are stored as WKB with GeoParquet 1.0 metadata, so that
DuckDB reads them as GEOMETRY.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
from pathlib import Path

from nominatim_api.sql.duckdb_layout import BBOX_TABLES, PARQUET_MAX_ROWS

from ..errors import UsageError

# pyarrow version with bloom filters (ParquetWriter bloom_filter_options).
PYARROW_MIN_VERSION = 24

# Column with the bytes of the heavy columns of a row while reading.
BYTES_COLUMN = '__nominatim_rowbytes'

# Rows read from DuckDB at once.
BATCH_SIZE = 8192

GEOPARQUET_TYPES = {'POINT': 'Point', 'LINESTRING': 'LineString', 'POLYGON': 'Polygon',
                    'MULTIPOINT': 'MultiPoint', 'MULTILINESTRING': 'MultiLineString',
                    'MULTIPOLYGON': 'MultiPolygon',
                    'GEOMETRYCOLLECTION': 'GeometryCollection'}


def import_pyarrow() -> Any:
    """ Import pyarrow, which is only needed for the conversion.
    """
    hint = f"Install it with: pip install 'pyarrow>={PYARROW_MIN_VERSION}'"
    try:
        import pyarrow  # type: ignore[import-not-found,import-untyped,unused-ignore]
        import pyarrow.parquet  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E501,F401
    except ImportError as err:
        raise UsageError("Converting to Parquet needs the Python package 'pyarrow'"
                         f" (>= {PYARROW_MIN_VERSION}). {hint}") from err
    if int(pyarrow.__version__.split('.')[0]) < PYARROW_MIN_VERSION:
        raise UsageError(f"Converting to Parquet needs pyarrow >= {PYARROW_MIN_VERSION},"
                         f" found {pyarrow.__version__}. {hint}")
    return pyarrow


class _RowGroupWriter:
    """ Collects record batches and writes them as row groups of at most
        `max_rows` rows, which are closed early when the heavy columns of
        their rows reach `max_bytes`.
    """

    def __init__(self, pa: Any, writer: Any, schema: Any,
                 max_rows: int, max_bytes: Optional[int]) -> None:
        self.pa = pa
        self.writer = writer
        self.schema = schema
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.batches: List[Any] = []
        self.rows = 0
        self.nbytes = 0
        self.groups = 0

    def add(self, batch: Any, sizes: Optional[Sequence[Optional[int]]]) -> None:
        start = 0
        while start < batch.num_rows:
            end = self._end_of_group(batch.num_rows, start, sizes)
            self.batches.append(batch.slice(start, end - start))
            start = end
            if self.rows >= self.max_rows \
               or (self.max_bytes is not None and self.nbytes >= self.max_bytes):
                self.flush()

    def _end_of_group(self, num_rows: int, start: int,
                      sizes: Optional[Sequence[Optional[int]]]) -> int:
        """ Count rows from `start` until the current row group is full
            or the batch ends. Return the end of the slice.
        """
        if sizes is None or self.max_bytes is None:
            end = min(num_rows, start + self.max_rows - self.rows)
            self.rows += end - start
            return end
        for i in range(start, num_rows):
            self.rows += 1
            self.nbytes += sizes[i] or 0
            if self.rows >= self.max_rows or self.nbytes >= self.max_bytes:
                return i + 1
        return num_rows

    def flush(self) -> None:
        if self.batches:
            table = self.pa.Table.from_batches(self.batches, schema=self.schema)
            self.writer.write_table(table, row_group_size=table.num_rows)
            self.groups += 1
        self.batches = []
        self.rows = 0
        self.nbytes = 0


def _geo_metadata(con: Any, source_sql: str, primary: Optional[str],
                  geoms: Sequence[str]) -> bytes:
    """ Return the GeoParquet 1.0 metadata for the given geometry columns.
    """
    columns: Dict[str, Any] = {}
    for col in geoms:
        types, xmin, ymin, xmax, ymax = con.execute(
            f"""SELECT list_distinct(list(ST_GeometryType("{col}")::VARCHAR)),
                       min(ST_XMin("{col}")), min(ST_YMin("{col}")),
                       max(ST_XMax("{col}")), max(ST_YMax("{col}"))
                  FROM ({source_sql})""").fetchone()
        meta: Dict[str, Any] = {'encoding': 'WKB',
                                'geometry_types': sorted(GEOPARQUET_TYPES[t] for t in types or []
                                                         if t in GEOPARQUET_TYPES)}
        if xmin is not None:
            meta['bbox'] = [xmin, ymin, xmax, ymax]
        columns[col] = meta

    return json.dumps({'version': '1.0.0',
                       'primary_column': primary if primary in geoms else geoms[0],
                       'columns': columns}).encode('utf-8')


def export_table(con: Any, name: str, source_sql: str, outfile: Path,
                 budget: Optional[Tuple[str, int]],
                 max_rows: int = PARQUET_MAX_ROWS,
                 bloom_filters: Sequence[str] = ()) -> int:
    """ Write the rows of `source_sql` in their physical order to the
        Parquet file `outfile`. `budget` is the (SQL expression, bytes)
        pair from PARQUET_ROWGROUP_BYTES or None, `max_rows` the row
        limit of a row group and `bloom_filters` the columns that get
        Parquet bloom filters (with pyarrow's defaults). Returns the
        number of row groups.
    """
    pa = import_pyarrow()

    columns = con.execute(f'DESCRIBE {source_sql}').fetchall()
    geoms = [c[0] for c in columns if c[1].startswith('GEOMETRY')]
    select = ', '.join(f'ST_AsWKB("{c[0]}") AS "{c[0]}"' if c[0] in geoms else f'"{c[0]}"'
                       for c in columns)
    expr = budget[0] if budget else '0'
    # Before the reader is opened: another query on the connection
    # would end the result stream of the reader.
    geo = _geo_metadata(con, source_sql, BBOX_TABLES.get(name), geoms) if geoms else None

    con.execute('SET arrow_lossless_conversion = true')  # keeps the JSON type
    reader = con.execute(f'SELECT {select}, {expr} AS {BYTES_COLUMN} FROM ({source_sql})')\
                .to_arrow_reader(BATCH_SIZE)
    schema = reader.schema.remove(reader.schema.get_field_index(BYTES_COLUMN))
    # With arrow_lossless_conversion, BOOLEAN comes as the extension type
    # arrow.bool8, which pyarrow would write as INT8. Write plain booleans.
    bools = [i for i, field in enumerate(schema) if field.type == pa.bool8()]
    for i in bools:
        schema = schema.set(i, schema.field(i).with_type(pa.bool_()))
    if geo is not None:
        schema = schema.with_metadata({b'geo': geo})

    writer = pa.parquet.ParquetWriter(str(outfile), schema, compression='zstd',
                                      bloom_filter_options={c: True for c in bloom_filters})
    try:
        groups = _RowGroupWriter(pa, writer, schema, max_rows, budget[1] if budget else None)
        for batch in reader:
            sizes = batch.column(BYTES_COLUMN).to_pylist() if budget else None
            batch = batch.drop_columns([BYTES_COLUMN])
            for i in bools:
                batch = batch.set_column(i, schema.field(i), batch.column(i).cast(pa.bool_()))
            groups.add(batch, sizes)
        groups.flush()
    finally:
        writer.close()

    return groups.groups
