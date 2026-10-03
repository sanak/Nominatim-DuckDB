# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2025 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Implementation of a category search around a place.
"""
from typing import Any, List, Tuple

import sqlalchemy as sa

from . import base
from ...typing import SaBind, SaFromClause
from ...types import SearchDetails, Bbox
from ...connection import SearchConnection
from ...sql.duckdb_layout import placex_rowids, placex_rowid
from ... import results as nres
from ..db_search_fields import WeightedCategories


LIMIT_PARAM: SaBind = sa.bindparam('limit')
MIN_RANK_PARAM: SaBind = sa.bindparam('min_rank')
MAX_RANK_PARAM: SaBind = sa.bindparam('max_rank')
COUNTRIES_PARAM: SaBind = sa.bindparam('countries')


class NearSearch(base.AbstractSearch):
    """ Category search of a place type near the result of another search.
    """
    def __init__(self, penalty: float, categories: WeightedCategories,
                 search: base.AbstractSearch) -> None:
        super().__init__(penalty)
        self.search = search
        self.categories = categories

    async def lookup(self, conn: SearchConnection,
                     details: SearchDetails) -> nres.SearchResults:
        """ Find results for the search in the database.
        """
        results = nres.SearchResults()
        base = await self.search.lookup(conn, details)

        if not base:
            return results

        base.sort(key=lambda r: (r.accuracy, r.rank_search))
        max_accuracy = base[0].accuracy + 0.5
        if base[0].rank_address == 0:
            min_rank = 0
            max_rank = 0
        elif base[0].rank_address < 26:
            min_rank = 1
            max_rank = min(25, base[0].rank_address + 4)
        else:
            min_rank = 26
            max_rank = 30
        base = nres.SearchResults(r for r in base
                                  if (r.source_table == nres.SourceTable.PLACEX
                                      and r.accuracy <= max_accuracy
                                      and r.bbox and r.bbox.area < 20
                                      and r.rank_address >= min_rank
                                      and r.rank_address <= max_rank))

        if base:
            baseids = [b.place_id for b in base[:5] if b.place_id]

            for category, penalty in self.categories:
                await self.lookup_category(results, conn, baseids, category, penalty, details)
                if len(results) >= details.max_results:
                    break

        return results

    async def lookup_category(self, results: nres.SearchResults,
                              conn: SearchConnection, ids: List[int],
                              category: Tuple[str, str], penalty: float,
                              details: SearchDetails) -> None:
        """ Find places of the given category near the list of
            place ids and add the results to 'results'.
        """
        table = conn.t.placex
        is_duckdb = conn.connection.dialect.name == 'duckdb'
        tgeom: SaFromClause
        if is_duckdb:
            # Read the base places from placex by row id (see duckdb_layout).
            rids = placex_rowids()
            tgeom = sa.select(table.c.place_id, table.c.rank_address,
                              table.c.geometry, table.c.centroid)\
                      .join_from(table, rids, placex_rowid() == rids.c.rid)\
                      .where(rids.c.place_id.in_(ids))\
                      .subquery('pgeom')
        else:
            tgeom = conn.t.placex.alias('pgeom')

        # Look up places of the category near the base place. The centroid
        # containment is served by the combined centroid/categories index.
        # We can afford to use a larger radius for the lookup.
        search_area = sa.case((sa.and_(tgeom.c.rank_address > 9,
                                       tgeom.c.geometry.is_area()),
                               tgeom.c.geometry),
                              else_=tgeom.c.centroid.ST_Expand(0.05))
        dist = sa.func.min(tgeom.c.centroid.ST_Distance(table.c.centroid)).label('dist')
        sql = sa.select(table.c.place_id, dist)\
                .join(tgeom, table.c.centroid.ST_CoveredBy(search_area))\
                .where(base.category_filter(table, *category))

        sql = sql.where(tgeom.c.place_id.in_(ids))

        t = conn.t.placex
        filters: List[Any] = \
            [base.no_index(t.c.rank_address).between(MIN_RANK_PARAM, MAX_RANK_PARAM)]
        restriction = base.category_restriction(t, details)
        if restriction is not None:
            filters.append(restriction)
        if details.countries:
            filters.append(t.c.country_code.in_(COUNTRIES_PARAM))
        if details.excluded:
            filters.append(base.exclude_places(t))
        if details.layers is not None:
            filters.append(base.filter_by_layer(t, details.layers))

        if is_duckdb:
            # The filters apply to the same row of placex inside and outside
            # (place_id is unique). Filter and limit the places inside, so
            # that only the full rows of the results are read, by row id.
            for where in filters:
                sql = sql.where(where)
            inner = sql.add_columns(placex_rowid().label('rid'))\
                       .group_by(table.c.place_id, placex_rowid())\
                       .order_by(dist).limit(LIMIT_PARAM).subquery()
            sql = base.select_placex(t).add_columns((-inner.c.dist).label('importance'))\
                      .join_from(t, inner, sa.and_(inner.c.place_id == t.c.place_id,
                                                   inner.c.rid == placex_rowid()))\
                      .order_by(inner.c.dist)
        else:
            inner = sql.group_by(table.c.place_id).subquery()
            sql = base.select_placex(t).add_columns((-inner.c.dist).label('importance'))\
                                       .join(inner, inner.c.place_id == t.c.place_id)\
                                       .order_by(inner.c.dist)
            for where in filters:
                sql = sql.where(where)

        sql = sql.limit(LIMIT_PARAM)

        bind_params = {'limit': details.max_results,
                       'min_rank': details.min_rank,
                       'max_rank': details.max_rank,
                       'excluded': details.excluded_place_ids,
                       'countries': details.countries}
        for row in await conn.execute(sql, bind_params):
            result = nres.create_from_placex_row(row, nres.SearchResult)
            result.accuracy = self.penalty + penalty
            result.bbox = Bbox.from_wkb(row.bbox)
            results.append(result)
