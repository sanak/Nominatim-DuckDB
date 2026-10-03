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

## Limitations

* The database is read-only. Updates are not possible; convert the
  PostgreSQL database again to get newer data.
* Geometry output in KML format (`polygon_kml`) is not supported and
  results in an error.
