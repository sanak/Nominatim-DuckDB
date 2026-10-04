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


@pytest.mark.parametrize('extra,keep', [([], False), (['--keep-duckdb'], True)])
def test_convert_format_parquet(cli_call, monkeypatch, tmp_path, extra, keep):
    pytest.importorskip('duckdb')
    tools = pytest.importorskip('nominatim_db.tools.convert_parquet')
    calls = []

    async def _convert(project_dir, outdir, options, keep_duckdb=False):
        calls.append((outdir, options, keep_duckdb))

    monkeypatch.setattr(tools, 'convert', _convert)
    outdir = tmp_path / 'pq'

    assert cli_call('convert', '--format', 'parquet', '-o', str(outdir),
                    '--without-details', *extra) == 0
    assert calls == [(outdir, {'reverse', 'search'}, keep)]


def test_convert_parquet_accepts_empty_directory(cli_call, monkeypatch, tmp_path):
    pytest.importorskip('duckdb')
    tools = pytest.importorskip('nominatim_db.tools.convert_parquet')

    async def _convert(*args, **kwargs):
        pass

    monkeypatch.setattr(tools, 'convert', _convert)
    outdir = tmp_path / 'pq'
    outdir.mkdir()

    assert cli_call('convert', '--format', 'parquet', '-o', str(outdir)) == 0


def test_convert_parquet_refuses_non_empty_directory(cli_call, tmp_path):
    outdir = tmp_path / 'pq'
    outdir.mkdir()
    (outdir / 'x').write_text('x')

    assert cli_call('convert', '--format', 'parquet', '-o', str(outdir)) == 1
