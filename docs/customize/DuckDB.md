A Nominatim database can be converted into a DuckDB database and used as
a read-only source for geocoding queries. This sections describes how to
create and use a DuckDB database.

!!! danger
    This feature is in an experimental state at the moment. Use at your own
    risk.

## Installing prerequisites

To use a DuckDB database, you need to install the Python packages
`duckdb` (>= 1.1), `duckdb-engine` and `pytz`. They are available as the
`duckdb` extra of the `nominatim-api` package. When installing from the
source tree, run:

    /srv/nominatim-venv/bin/pip install 'packaging/nominatim-api[duckdb]'

Nominatim also needs the DuckDB `spatial` extension. It never downloads
extensions by itself, so the extension must be installed beforehand with
the same DuckDB version that Nominatim uses:

    /srv/nominatim-venv/bin/python -c "import duckdb; duckdb.connect().install_extension('spatial')"

This installs the extension into DuckDB's default extension directory
(`~/.duckdb/extensions`) of the current user. If the database is used by
a different user, for example the user of the webserver, install the
extension into a directory readable by that user and point
[NOMINATIM_DUCKDB_EXTENSION_DIR](Settings.md#nominatim_duckdb_extension_dir)
to it:

    /srv/nominatim-venv/bin/python -c "import duckdb; duckdb.connect(config={'extension_directory': '/srv/duckdb-extensions'}).install_extension('spatial')"

## Creating a new DuckDB database

Nominatim cannot import directly into a DuckDB database. Instead you have to
first create a geocoding database in PostgreSQL by running a
[regular Nominatim import](../admin/Import.md).

Once this is done, the database can be converted to DuckDB with

    nominatim convert --format duckdb -o mydb.duckdb

This will create a database where all geocoding functions are available.
Use `--without-search` to leave out the tables needed for forward search
and make the database smaller. The switches `--without-reverse` and
`--without-details` are accepted but currently have no effect.

The conversion stores intermediate files next to the output file, so make
sure that there is enough free disk space in that directory. The
environment variables `DUCKDB_MEMORY_LIMIT` and `DUCKDB_TEMP_DIR` can be
used to restrict the memory DuckDB uses during the conversion and to set
the directory where it writes data that does not fit into memory.

### Creating Parquet files

The database can also be exported as a directory with one Parquet file
per table:

    nominatim convert --format parquet -o /srv/nominatim-parquet/

The Parquet export needs the Python package `pyarrow` (>= 24) during the
conversion only. The output directory must be new or empty. A DuckDB
database is created in it first and removed at the end; add
`--keep-duckdb` to keep it as `nominatim.duckdb`, for example to serve
the same data as a database file. During the conversion, the directory
needs space for both the DuckDB database and the Parquet files.

## Using a DuckDB database

Once you have created the database, you can use it by simply pointing the
database DSN to the DuckDB file:

    NOMINATIM_DATABASE_DSN=duckdb:dbname=mydb.duckdb

DuckDB support is only available for the Python frontend. The CLI query
commands, the library interface and `nominatim serve` work right out of
the box.

The physical layout of the DuckDB database depends on the version of
Nominatim. When Nominatim is updated to a version with a different layout,
it refuses to open older database files with an error message. Create
the database again with `nominatim convert` in that case.

### Remote databases and Parquet files

A database file or a Parquet export can also be read from object storage
or a web server. The DuckDB extension `httpfs` must be installed
beforehand in the same way as `spatial`:

    /srv/nominatim-venv/bin/python -c "import duckdb; duckdb.connect().install_extension('httpfs')"

Point the DSN to the remote database file or to the Parquet directory:

    NOMINATIM_DATABASE_DSN=duckdb:dbname=s3://my-bucket/nominatim/2026-10-04/nominatim.duckdb
    NOMINATIM_DATABASE_DSN=duckdb:parquet=s3://my-bucket/nominatim/2026-10-04/

`parquet=` also accepts a local directory. Credentials for S3 are taken
from the usual AWS environment variables (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`). For other
setups, for example S3-compatible storage, create a DuckDB secret with
[NOMINATIM_DUCKDB_INIT_SQL](Settings.md#nominatim_duckdb_init_sql):

    NOMINATIM_DUCKDB_INIT_SQL="CREATE SECRET (TYPE s3, KEY_ID '...', SECRET '...', ENDPOINT 'storage.example.com', URL_STYLE 'path')"

Remote data is cached in memory and never checked for changes. Never
replace files in place: put a new version of the data under a new path
and change the DSN.

The first queries after a start have to fetch data from the remote
storage and are much slower than later queries. Parquet exports usually
transfer less data than a remote database file. With a remote database
file, single queries right after a start can take tens of seconds (up
to about 30 s for Japan). Raise
[NOMINATIM_QUERY_TIMEOUT](Settings.md#nominatim_query_timeout)
accordingly or use a Parquet export instead.

With a Parquet export, the table `search_name` is copied into memory
when the frontend opens its first connection, because the search is
much faster that way. This needs additional memory: loading it raises
the peak memory use to about 4 GB for Japan. It also adds the time to
read it to the start-up (for Japan about 4 s from S3, about 1.5 s from
a local directory). Do not set the DuckDB `memory_limit` (for example
with `NOMINATIM_DUCKDB_INIT_SQL`) below the size of the table.

## Limitations

* The database is read-only. Updates are not possible; convert the
  PostgreSQL database again to get newer data.
* Remote databases and Parquet files are read-only, like local databases.
* Geometry output in KML format (`polygon_kml`) is not supported and
  results in an error.
