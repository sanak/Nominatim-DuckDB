# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2026 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Tests for the 'convert' subcommand.
"""
import pytest


@pytest.mark.parametrize('fmt,module', [('sqlite', 'convert_sqlite'),
                                        ('duckdb', 'convert_duckdb')])
def test_convert_format(cli_call, monkeypatch, tmp_path, fmt, module):
    if fmt == 'duckdb':
        pytest.importorskip('duckdb')
    tools = pytest.importorskip(f'nominatim_db.tools.{module}')
    calls = []

    async def _convert(project_dir, outfile, options):
        calls.append((outfile, options))

    monkeypatch.setattr(tools, 'convert', _convert)
    outfile = tmp_path / f'out.{fmt}'

    assert cli_call('convert', '--format', fmt, '-o', str(outfile),
                    '--without-details') == 0
    assert calls == [(outfile, {'reverse', 'search'})]


def test_convert_refuses_existing_file(cli_call, tmp_path):
    outfile = tmp_path / 'out.duckdb'
    outfile.write_text('x')

    assert cli_call('convert', '--format', 'duckdb', '-o', str(outfile)) == 1
