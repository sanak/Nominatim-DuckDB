# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of Nominatim. (https://nominatim.org)
#
# Copyright (C) 2024 by the Nominatim developer community.
# For a full list of authors see the git log.
"""
Implementation of the 'convert' subcommand.
"""
from typing import Set, Any, Union, Optional, Sequence
import argparse
from pathlib import Path

from ..errors import UsageError
from .args import NominatimArgs
from ..utils.asyncio_utils import asyncio_run


class WithAction(argparse.Action):
    """ Special action that saves a list of flags, given on the command-line
        as `--with-foo` or `--without-foo`.
    """
    def __init__(self, option_strings: Sequence[str], dest: Any,
                 default: bool = True, **kwargs: Any) -> None:
        if 'nargs' in kwargs:
            raise ValueError("nargs not allowed.")
        if option_strings is None:
            raise ValueError("Positional parameter not allowed.")

        self.dest_set = kwargs.pop('dest_set')
        full_option_strings = []
        for opt in option_strings:
            if not opt.startswith('--'):
                raise ValueError("short-form options not allowed")
            if default:
                self.dest_set.add(opt[2:])
            full_option_strings.append(f"--with-{opt[2:]}")
            full_option_strings.append(f"--without-{opt[2:]}")

        super().__init__(full_option_strings, argparse.SUPPRESS, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: Optional[str] = None) -> None:
        assert option_string
        if option_string.startswith('--with-'):
            self.dest_set.add(option_string[7:])
        if option_string.startswith('--without-'):
            self.dest_set.discard(option_string[10:])


class ConvertDB:
    """ Convert an existing database into a different format. (EXPERIMENTAL)

        Dump a read-only version of the database in a different format.
        Supported are SQLite and DuckDB databases and Parquet files.
    """

    def __init__(self) -> None:
        self.options: Set[str] = set()

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--format', default='sqlite',
                            choices=('sqlite', 'duckdb', 'parquet'),
                            help='Format of the output database (default: sqlite)')
        parser.add_argument('--output', '-o', required=True, type=Path,
                            help='File to write the database to'
                                 ' (an empty or new directory for parquet).')
        parser.add_argument('--keep-duckdb', action='store_true',
                            help='parquet only: keep the intermediate DuckDB database'
                                 ' in the output directory')
        group = parser.add_argument_group('Switches to define database layout'
                                          '(currently no effect)')
        group.add_argument('--reverse', action=WithAction, dest_set=self.options, default=True,
                           help='Enable/disable support for reverse and lookup API'
                                ' (default: enabled)')
        group.add_argument('--search', action=WithAction, dest_set=self.options, default=True,
                           help='Enable/disable support for search API (default: disabled)')
        group.add_argument('--details', action=WithAction, dest_set=self.options, default=True,
                           help='Enable/disable support for details API (default: enabled)')

    def run(self, args: NominatimArgs) -> int:
        if args.format == 'parquet':
            if args.output.exists() and (not args.output.is_dir()
                                         or any(args.output.iterdir())):
                raise UsageError(f"Output directory '{args.output}' exists and is not empty.")
            from ..tools import convert_parquet
            asyncio_run(convert_parquet.convert(args.project_dir, args.output, self.options,
                                                keep_duckdb=args.keep_duckdb))
            return 0

        if args.output.exists():
            raise UsageError(f"File '{args.output}' already exists. Refusing to overwrite.")

        if args.format == 'sqlite':
            from ..tools import convert_sqlite
            asyncio_run(convert_sqlite.convert(args.project_dir, args.output, self.options))
            return 0

        if args.format == 'duckdb':
            from ..tools import convert_duckdb
            asyncio_run(convert_duckdb.convert(args.project_dir, args.output, self.options))
            return 0

        return 1
