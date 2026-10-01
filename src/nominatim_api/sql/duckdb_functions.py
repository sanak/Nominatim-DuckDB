# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Custom functions for DuckDB.
"""
from typing import Any


def install_custom_functions(conn: Any) -> None:
    """ Install helper functions for Nominatim into the given DuckDB
        DBAPI connection. There are no custom functions yet.
    """
