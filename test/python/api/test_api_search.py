# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2025 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for search API calls.

These tests make sure that all Python code is correct and executable.
Functional tests can be found in the BDD test suite.
"""
import pytest
import sqlalchemy as sa

import nominatim_api as napi
import nominatim_api.logging as loglib

API_OPTIONS = {'search'}


@pytest.fixture(autouse=True)
def setup_icu_tokenizer(apiobj):
    """ Setup the properties needed for using the ICU tokenizer.
    """
    apiobj.add_data('properties',
                    [{'property': 'tokenizer', 'value': 'icu'},
                     {'property': 'tokenizer_import_normalisation', 'value': ':: lower();'},
                     {'property': 'tokenizer_import_transliteration',
                      'value': "'1' > '/1/'; 'ä' > 'ä '"},
                     ])


def test_search_no_content(apiobj, frontend):
    apiobj.add_word_table([])

    api = frontend(apiobj, options=API_OPTIONS)
    assert api.search('foo') == []


def test_search_simple_word(apiobj, frontend):
    apiobj.add_word_table([(55, 'test', 'W', 'test', None),
                           (2, 'test', 'w', 'test', None)])

    apiobj.add_placex(place_id=444, class_='place', type='village',
                      centroid=(1.3, 0.7))
    apiobj.add_search_name(444, names=[2, 55])

    api = frontend(apiobj, options=API_OPTIONS)
    results = api.search('TEST')

    assert [r.place_id for r in results] == [444]


def test_search_japanese_name(apiobj, frontend):
    apiobj.add_word_table([(101, '東京駅', 'W', '東京駅', None),
                           (102, '東京駅', 'w', '東京駅', None),
                           (103, '丸の内', 'W', '丸の内', None),
                           (104, '丸の内', 'w', '丸の内', None)])

    apiobj.add_placex(place_id=500, class_='railway', type='station',
                      name={'name': '東京駅', 'name:en': 'Tokyo Station'},
                      centroid=(139.7671, 35.6812))
    apiobj.add_search_name(500, names=[101, 102], address=[103, 104],
                           centroid=(139.7671, 35.6812))
    apiobj.add_placex(place_id=501, class_='place', type='quarter', name='丸の内',
                      rank_search=20, rank_address=20,
                      centroid=(139.7638, 35.6825))
    apiobj.add_search_name(501, names=[103, 104], search_rank=20, address_rank=20,
                           centroid=(139.7638, 35.6825))

    api = frontend(apiobj, options=API_OPTIONS)

    results = api.search('東京駅')
    assert [r.place_id for r in results] == [500]
    assert results[0].names['name'] == '東京駅'

    assert [r.place_id for r in api.search('東京駅, 丸の内')] == [500]
    assert [r.place_id for r in api.search('丸の内')][0] == 501

    details = api.details(napi.PlaceID(500), keywords=True)
    assert {w.word for w in details.name_keywords} == {'東京駅'}
    assert {w.word for w in details.address_keywords} == {'丸の内'}


REGEXP_WORD_CASES = [('1番地', '1', False), ('東京駅1', '1', False),
                     ('丁目3', '3', False), ('ä1', '1', False),
                     ('１', '１', True), ('1番地', '1番地', True),
                     ('12a', '12|3', False), ('12a', '12a|3', True),
                     ('112', '12', False)]


def test_regexp_word_same_on_all_backends(apiobj, frontend):
    """ Housenumber matching must use Unicode word boundaries everywhere.
    """
    apiobj.add_word_table([])
    api = frontend(apiobj, options=API_OPTIONS)

    async def _match(text, words):
        async with api._async_api.begin() as conn:
            return await conn.scalar(sa.select(sa.func.RegexpWord(words, sa.literal(text))))

    for text, words, expected in REGEXP_WORD_CASES:
        assert bool(api._loop.run_until_complete(_match(text, words))) == expected, (text, words)


@pytest.mark.parametrize('logtype', ['text', 'html'])
def test_search_with_debug(apiobj, frontend, logtype):
    apiobj.add_word_table([(55, 'test', 'W', 'test', None),
                           (2, 'test', 'w', 'test', None)])

    apiobj.add_placex(place_id=444, class_='place', type='village',
                      centroid=(1.3, 0.7))
    apiobj.add_search_name(444, names=[2, 55])

    api = frontend(apiobj, options=API_OPTIONS)
    loglib.set_log_output(logtype)
    api.search('TEST')

    assert loglib.get_and_disable()


def test_address_no_content(apiobj, frontend):
    apiobj.add_word_table([])

    api = frontend(apiobj, options=API_OPTIONS)
    assert api.search_address(amenity='hotel',
                              street='Main St 34',
                              city='Happyville',
                              county='Wideland',
                              state='Praerie',
                              postalcode='55648',
                              country='xx') == []


@pytest.mark.parametrize('atype,address,search', [('street', 26, 26),
                                                  ('city', 16, 18),
                                                  ('county', 12, 12),
                                                  ('state', 8, 8)])
def test_address_simple_places(apiobj, frontend, atype, address, search):
    apiobj.add_word_table([(55, 'test', 'W', 'test', None),
                           (2, 'test', 'w', 'test', None)])

    apiobj.add_placex(place_id=444,
                      rank_address=address, rank_search=search,
                      centroid=(1.3, 0.7))
    apiobj.add_search_name(444, names=[2, 55], address_rank=address, search_rank=search)

    api = frontend(apiobj, options=API_OPTIONS)
    results = api.search_address(**{atype: 'TEST'})

    assert [r.place_id for r in results] == [444]


def test_address_country(apiobj, frontend):
    apiobj.add_word_table([(None, 'ro', 'C', 'ro', {'cc': 'ro'})])
    apiobj.add_country('ro', 'POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))')
    apiobj.add_country_name('ro', {'name': 'România'})

    api = frontend(apiobj, options=API_OPTIONS)
    assert len(api.search_address(country='ro')) == 1


def test_category_no_categories(apiobj, frontend):
    apiobj.add_word_table([])

    api = frontend(apiobj, options=API_OPTIONS)
    assert api.search_category([], near_query='Berlin') == []


def test_category_no_content(apiobj, frontend):
    apiobj.add_word_table([])

    api = frontend(apiobj, options=API_OPTIONS)
    assert api.search_category([('amenity', 'restaurant')]) == []


def test_category_simple_restaurant(apiobj, frontend):
    apiobj.add_word_table([])

    apiobj.add_placex(place_id=444, class_='amenity', type='restaurant',
                      centroid=(1.3, 0.7))
    apiobj.add_search_name(444, names=[2, 55], address_rank=16, search_rank=18)

    api = frontend(apiobj, options=API_OPTIONS)
    results = api.search_category([('amenity', 'restaurant')],
                                  near=(1.3, 0.701), near_radius=0.015)

    assert [r.place_id for r in results] == [444]


def test_category_with_search_phrase(apiobj, frontend):
    apiobj.add_word_table([(55, 'test', 'W', 'test', None),
                           (2, 'test', 'w', 'test', None)])

    apiobj.add_placex(place_id=444, class_='place', type='village',
                      rank_address=16, rank_search=18,
                      centroid=(1.3, 0.7))
    apiobj.add_search_name(444, names=[2, 55], address_rank=16, search_rank=18)
    apiobj.add_placex(place_id=95, class_='amenity', type='restaurant',
                      centroid=(1.3, 0.7003))

    api = frontend(apiobj, options=API_OPTIONS)
    results = api.search_category([('amenity', 'restaurant')], near_query='TEST')

    assert [r.place_id for r in results] == [95]
