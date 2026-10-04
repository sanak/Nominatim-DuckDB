# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Data sources of the DuckDB frontend.

The part of a DuckDB DSN after `duckdb:` names one of:

  * a local database file (`dbname=<path>`), opened directly read-only,
  * a remote database file (`dbname=<url>`, e.g. `s3://` or `https://`),
    attached read-only through the `httpfs` extension,
  * a directory or URL prefix with one Parquet file per table
    (`parquet=<path or url>`), as written by
    `nominatim convert --format parquet`.

Remote and Parquet sources are served from a named in-memory DuckDB
instance, see `DuckDBSource.instance_name()`.
"""
from typing import Any, Literal
import dataclasses
import hashlib

from ..errors import UsageError

SourceKind = Literal['file', 'remote', 'parquet']


def is_url(location: str) -> bool:
    """ Check if the location is a URL (with a scheme like s3:// or https://).
    """
    return '://' in location


@dataclasses.dataclass(frozen=True)
class DuckDBSource:
    """ Kind and location of a DuckDB data source.
    """
    kind: SourceKind
    location: str

    def instance_name(self, extension_dir: str) -> str:
        """ Return the name of the in-memory DuckDB instance for a remote
            or Parquet source. All connections opened with the same name
            in one process share the instance and with it the buffer pool
            and the cache of remote files. DuckDB refuses to open an
            instance twice with a different configuration, so the name
            depends on the extension directory, too.
        """
        key = f'{self.kind}\0{self.location}\0{extension_dir}'.encode('utf-8')
        return f':memory:nominatim_{hashlib.sha256(key).hexdigest()[:16]}'

    def parquet_file(self, table: str) -> str:
        """ Return the location of the Parquet file for the given table.
        """
        return f"{self.location.rstrip('/')}/{table}.parquet"


def parse_dsn(dsn: str) -> DuckDBSource:
    """ Parse the part of a DuckDB DSN after the 'duckdb:' prefix.
    """
    params = dict(p.split('=', 1) for p in dsn.split(';') if p)
    if 'dbname' in params and 'parquet' in params:
        raise UsageError("A DuckDB DSN must have either 'dbname' or 'parquet', not both.")
    if 'parquet' in params:
        return DuckDBSource('parquet', params['parquet'])

    dbname = params.get('dbname', '')
    return DuckDBSource('remote' if is_url(dbname) else 'file', dbname)


# Name under which a remote database file is attached.
REMOTE_ALIAS = 'nominatim_remote'

# Settings for remote and Parquet sources. The data is never changed in
# place (a new version gets a new location), so cached remote files and
# Parquet metadata do not need to be validated again.
CACHE_SETTINGS = ("SET validate_external_file_cache = 'NO_VALIDATION'",
                  'SET parquet_metadata_cache = true')

# Settings that only exist once httpfs is loaded.
HTTP_CACHE_SETTINGS = ('SET enable_http_metadata_cache = true', )


def quote(value: str) -> str:
    """ Quote a string literal for DuckDB.
    """
    return "'" + value.replace("'", "''") + "'"


class SourceConnector:
    """ Prepares new DuckDB connections for a data source: runs the
        configured init SQL and makes the tables of remote and Parquet
        sources available. Call `setup()` with a cursor of every new
        connection after the spatial extension has been loaded.
    """

    def __init__(self, source: DuckDBSource, init_sql: str) -> None:
        self.source = source
        self.init_sql = [s.strip() for s in init_sql.split(';') if s.strip()]

    def setup(self, cursor: Any) -> None:
        """ Initialise a new connection.
        """
        if self.source.kind == 'file':
            self._run_init_sql(cursor)
            return

        if is_url(self.source.location):
            cursor.execute('LOAD httpfs')
            for sql in HTTP_CACHE_SETTINGS:
                cursor.execute(sql)
        for sql in CACHE_SETTINGS:
            cursor.execute(sql)
        self._run_init_sql(cursor)

        if self.source.kind == 'remote':
            cursor.execute(f'ATTACH IF NOT EXISTS {quote(self.source.location)}'
                           f' AS {REMOTE_ALIAS} (READ_ONLY)')
            cursor.execute(f'USE {REMOTE_ALIAS}')

    def _run_init_sql(self, cursor: Any) -> None:
        for sql in self.init_sql:
            cursor.execute(sql)
