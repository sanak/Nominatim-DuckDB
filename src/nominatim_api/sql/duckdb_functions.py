# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Custom functions for DuckDB.

There are none: the helper functions that SQLite implements in Python
(weigh_search, array_contains and the array aggregates) are expressed
in plain SQL with the native list functions of DuckDB instead, see
`duckdb_compilers`. Python UDFs in DuckDB need numpy and cannot be
aggregates.
"""
from typing import Any


def install_custom_functions(conn: Any) -> None:
    """ Install helper functions for Nominatim into the given DuckDB
        DBAPI connection. Nothing to do, see the module documentation.
    """
