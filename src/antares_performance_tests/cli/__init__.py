# SPDX-FileCopyrightText: 2024-present Laurent LAPORTE <laurent.laporte.pro@gmail.com>
#
# SPDX-License-Identifier: MIT
import click

from antares_performance_tests.__about__ import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="antares-performance-tests")
def antares_performance_tests():
    click.echo("Hello world!")
