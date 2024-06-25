"""
The goal of this benchmark is to measure the saving times of a DataFrame,
using different backup methods (CSV, Arrow, HDF5, Excel).

Measurements will be made on annual time series with the usual time steps:
- 1 year (freq='Y')
- 52 weeks (freq='W')
- 365 days (freq='D')
- 8760 hours (freq='H')

The matrices will have 1, 10, 100 and 1000 columns.

For the CSV format, we'll choose different separators (comma, semicolon, tab).

Saving performance for the Excel format varies according to the engine used for serialization (openpyxl, xlsxwriter).

To compare performance, we'll select the results for matrices (8760, 1000) and
the following formats: CSV (semicolon), Arrow, HDF5, Excel (openpyxl, xlsxwriter).
"""

import argparse
import configparser
import dataclasses
import enum
import logging
import logging.config
import sys
import tempfile
import time
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

_REQUIRED_PACKAGES = {
    "openpyxl": "Benchmarks with XLSX format",
    "xlsxwriter": "Benchmarks with XLSX format",
    "tables": "Benchmarks with Arrow (or HDF5) format",
    "pyarrow": "Benchmarks with Arrow format",
    "tabulate": "Generate the benchmark report",
}

for package in _REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        raise ImportError(f"The '{package}' package is required for this benchmark") from None

logger = logging.getLogger("dataframe_saving_benchmark")


class EnumIgnoreCase(str, enum.Enum):
    """
    Case-insensitive enum base class

    Usage:

    >>> class WeekDay(EnumIgnoreCase):
    ...     MONDAY = "Monday"
    ...     TUESDAY = "Tuesday"
    ...     WEDNESDAY = "Wednesday"
    ...     THURSDAY = "Thursday"
    ...     FRIDAY = "Friday"
    ...     SATURDAY = "Saturday"
    ...     SUNDAY = "Sunday"
    >>> WeekDay("monday")
    <WeekDay.MONDAY: 'Monday'>
    >>> WeekDay("MONDAY")
    <WeekDay.MONDAY: 'Monday'>
    """

    @classmethod
    def _missing_(cls, value: object) -> t.Optional["EnumIgnoreCase"]:
        if isinstance(value, str):
            for member in cls:
                # noinspection PyUnresolvedReferences
                if member.value.upper() == value.upper():
                    # noinspection PyTypeChecker
                    return member
        # `value` is not a valid
        return None


class TableExportFormat(EnumIgnoreCase):
    """Export format for tables."""

    XLSX = "xlsx (openpyxl)"
    XLSX_WRITER = "xlsx (xlsxwriter)"
    ARROW = "arrow"
    HDF5 = "hdf5"
    TSV = "tsv"
    CSV = "csv (comma)"
    CSV_SEMICOLON = "csv (semicolon)"

    def __str__(self) -> str:
        """Return the format as a string for display."""
        return self.value.title()

    @property
    def media_type(self) -> str:
        """Return the media type used for the HTTP response."""
        if self in (TableExportFormat.XLSX, TableExportFormat.XLSX_WRITER):
            # noinspection SpellCheckingInspection
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif self == TableExportFormat.TSV:
            return "text/tab-separated-values"
        elif self in (TableExportFormat.CSV, TableExportFormat.CSV_SEMICOLON):
            return "text/csv"
        elif self == TableExportFormat.ARROW:
            return "application/vnd.apache.arrow.file"
        elif self == TableExportFormat.HDF5:
            return "application/x-hdf5"
        else:  # pragma: no cover
            raise NotImplementedError(f"Export format '{self}' is not implemented")

    @property
    def suffix(self) -> str:
        """Return the file suffix for the format."""
        if self in (TableExportFormat.XLSX, TableExportFormat.XLSX_WRITER):
            return ".xlsx"
        elif self == TableExportFormat.TSV:
            return ".tsv"
        elif self in (TableExportFormat.CSV, TableExportFormat.CSV_SEMICOLON):
            return ".csv"
        elif self == TableExportFormat.ARROW:
            return ".arrow"
        elif self == TableExportFormat.HDF5:
            return ".h5"
        else:  # pragma: no cover
            raise NotImplementedError(f"Export format '{self}' is not implemented")

    def export_table(
        self,
        df: pd.DataFrame,
        export_path: t.Union[str, Path],
        *,
        with_index: bool = True,
        with_header: bool = True,
    ) -> None:
        """Export a table to a file in the given format."""
        if self == TableExportFormat.XLSX:
            return df.to_excel(
                export_path,
                index=with_index,
                header=with_header,
                engine="openpyxl",
            )
        elif self == TableExportFormat.XLSX_WRITER:
            return df.to_excel(
                export_path,
                index=with_index,
                header=with_header,
                engine="xlsxwriter",
            )
        elif self == TableExportFormat.TSV:
            return df.to_csv(
                export_path,
                sep="\t",
                index=with_index,
                header=with_header,
                float_format="%.6f",
            )
        elif self == TableExportFormat.CSV:
            return df.to_csv(
                export_path,
                sep=",",
                index=with_index,
                header=with_header,
                float_format="%.6f",
            )
        elif self == TableExportFormat.CSV_SEMICOLON:
            return df.to_csv(
                export_path,
                sep=";",
                decimal=",",
                index=with_index,
                header=with_header,
                float_format="%.6f",
            )
        elif self == TableExportFormat.ARROW:
            # Feather does not directly support serializing a DatetimeIndex as an index.
            # We need to convert it to a column before saving the DataFrame.
            df_reset = df.reset_index()
            return df_reset.to_feather(export_path, compression="uncompressed")
        elif self == TableExportFormat.HDF5:
            return df.to_hdf(
                export_path,
                key="data",
                mode="w",
                format="fixed",  # "fixed" is faster than "table" for writing
                data_columns=True,
            )
        else:  # pragma: no cover
            raise NotImplementedError(f"Export format '{self}' is not implemented")


def measure_df_saving_time(
    tmp_dir: Path,
    export_format: TableExportFormat,
    *,
    columns=(1, 10, 100, 1000),
    frequencies=((1, "Y"), (52, "W"), (365, "D"), (8760, "H")),
    seed=777,
) -> pd.DataFrame:
    """Measure the time to save DataFrames with different dimensions and frequencies."""

    frequencies = dict(frequencies)

    saving_time_table = pd.DataFrame(
        np.zeros((len(frequencies), len(columns))),
        index=list(frequencies.values()),
        columns=columns,
    )

    generator = np.random.default_rng(seed)

    for i, col in enumerate(columns):
        for j, (freq, freq_str) in enumerate(frequencies.items()):
            # Prepare the DataFrame
            df = pd.DataFrame(
                generator.standard_normal((freq, col)),
                index=pd.date_range("2018-01-01", periods=freq, freq=freq_str),
                columns=[f"TS_{i}" for i in range(1, col + 1)],
            )

            # Measure the time to export the DataFrame
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir,
                prefix=f"~{freq_str}_{col}_",
                suffix=export_format.suffix,
                delete=True,
            ) as tf:
                export_path = Path(tf.name)
                logger.info(
                    f"- Exporting table with {col:-4d} columns and frequency {freq:-5d}{freq_str}"
                    f" to {export_format} format..."
                )
                start_time = time.time()
                export_format.export_table(df, export_path)
                duration = time.time() - start_time
                saving_time_table.loc[freq_str, col] = duration

    return saving_time_table


@dataclasses.dataclass
class BenchmarkConfig:
    columns: t.Sequence[int]
    frequencies: t.Mapping[int, str]
    seed: int

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "BenchmarkConfig":
        # remove spaces
        columns = [c.strip() for c in config["benchmark"]["columns"].split(",")]
        frequencies = [f.strip() for f in config["benchmark"]["frequencies"].split(",")]
        return cls(
            columns=[int(col) for col in columns],
            frequencies=dict((int(freq[:-1]), freq[-1]) for freq in frequencies),
            seed=config.getint("benchmark", "seed"),
        )


@dataclasses.dataclass
class ReportConfig:
    report_file: Path

    @classmethod
    def from_config(
        cls,
        config: configparser.ConfigParser,
        config_dir: Path,
    ) -> "ReportConfig":
        report_file = Path(config.get("report", "report_file")).expanduser()
        if not report_file.is_absolute():
            report_file = config_dir / report_file
        return cls(report_file=report_file)


@dataclasses.dataclass
class Config:
    benchmark: BenchmarkConfig
    report: ReportConfig

    @classmethod
    def from_config(cls, config: configparser.ConfigParser, config_dir: Path) -> "Config":
        return cls(
            benchmark=BenchmarkConfig.from_config(config),
            report=ReportConfig.from_config(config, config_dir),
        )


# Default `benchmark.ini` configuration file:
BENCHMARK_INI = """[benchmark]
columns = 1, 10, 100, 1000
frequencies = 1Y, 52W, 365D, 8760H
seed = 777

[report]
report_file = dataframe_saving_benchmark.md
"""

# Default `logging.ini` configuration file:
# noinspection SpellCheckingInspection
LOGGING_INI = """[loggers]
keys = root,dataframe_saving_benchmark

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = DEBUG
handlers = console

[logger_dataframe_saving_benchmark]
level = DEBUG
handlers = console
qualname = dataframe_saving_benchmark
propagate = 0

[handler_console]
class = StreamHandler
level = DEBUG
formatter = generic
args = (sys.stdout,)

[formatter_generic]
format = [%(asctime)s] %(levelname)s: %(message)s
datefmt =
"""


def build_saving_time_table_path(base_path: Path, export_format: TableExportFormat) -> Path:
    saving_time_table_name = f"{base_path.stem}-{export_format.name}{base_path.suffix}"
    return base_path.parent.joinpath(saving_time_table_name)


def calc_all_saving_times(base_path: Path, config: Config) -> None:
    columns = config.benchmark.columns
    frequencies = config.benchmark.frequencies

    export_format: TableExportFormat
    count = len(TableExportFormat)
    for step, export_format in enumerate(TableExportFormat, start=1):
        saving_time_table_path = build_saving_time_table_path(base_path, export_format)

        if saving_time_table_path.exists():
            # load the HDF5 table which key is the name of the export format
            try:
                saving_time_table = t.cast(pd.DataFrame, pd.read_feather(saving_time_table_path))
                saving_time_table.set_index("index", inplace=True)
                saving_time_table.columns = [int(c) for c in saving_time_table.columns]
            except FileNotFoundError:
                pass
            else:
                # fmt: off
                if (
                    saving_time_table.columns.equals(pd.Index(columns)) and
                    saving_time_table.index.equals(pd.Index(frequencies.values()))
                ):
                    logger.info(f"Skipping the benchmark for {export_format:18s}: already done [{step}/{count}]")
                    continue

        logger.info(f"Processing benchmark for {export_format:18s} [{step}/{count}]...")
        saving_time_table = measure_df_saving_time(
            base_path.parent,
            export_format,
            columns=columns,
            frequencies=frequencies.items(),
            seed=config.benchmark.seed,
        )

        saving_time_table.columns = [str(c) for c in saving_time_table.columns]
        saving_time_table.reset_index(inplace=True)
        saving_time_table.to_feather(saving_time_table_path)


def calc_fastest_exports(base_path: Path, columns: t.Sequence[int], frequencies: t.Mapping[int, str]) -> pd.DataFrame:
    # For the report, we compare the saving times of each format for the largest dimensions
    column = max(columns)
    freq = max(frequencies)
    freq_str = frequencies[freq]
    col_time = "time (s)"
    col_ratio = "ratio"
    fastest_exports = pd.DataFrame(columns=[col_time, col_ratio], index=[str(f) for f in TableExportFormat])
    export_format: TableExportFormat
    for export_format in TableExportFormat:
        saving_time_table_path = build_saving_time_table_path(base_path, export_format)
        saving_time_table = t.cast(pd.DataFrame, pd.read_feather(saving_time_table_path))
        saving_time_table.set_index("index", inplace=True)
        saving_time_table.columns = [int(c) for c in saving_time_table.columns]
        fastest_exports.loc[str(export_format), col_time] = saving_time_table.loc[freq_str, column].round(1)
    fastest_exports[col_ratio] = fastest_exports[col_time] / fastest_exports[col_time].min()
    fastest_exports.sort_values(col_time, inplace=True)
    return fastest_exports


def benchmark(config_dir: Path, *, clear_cache: bool = False) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)

    logging_ini_path = config_dir / "logging.ini"
    if not logging_ini_path.exists():
        logging_ini_path.write_text(LOGGING_INI)
    logging.config.fileConfig(logging_ini_path)

    benchmark_ini_path = config_dir / "benchmark.ini"
    if not benchmark_ini_path.exists():
        benchmark_ini_path.write_text(BENCHMARK_INI)
    benchmark_config = configparser.ConfigParser()
    benchmark_config.read(benchmark_ini_path)

    config = Config.from_config(benchmark_config, config_dir)
    columns = config.benchmark.columns
    frequencies = config.benchmark.frequencies

    # Several Arrow files to store the results of the benchmark
    if clear_cache:
        for path in config_dir.glob("saving_time_table*.arrow"):
            path.unlink()

    saving_time_table_base_path = config_dir / "saving_time_table.arrow"
    calc_all_saving_times(saving_time_table_base_path, config)
    fastest_exports = calc_fastest_exports(saving_time_table_base_path, columns, frequencies)

    # Generate the report (compare the results of the different formats)
    logger.info(f"Generating the benchmark report '{config.report.report_file}'...")
    with open(config.report.report_file, "w") as fh:
        fh.write("# DataFrame Saving Benchmark\n\n")
        fh.write("## Benchmark configuration\n\n")
        fh.write("```ini\n")
        benchmark_config.write(fh)
        fh.write("```\n\n")
        fh.write("## Benchmark results\n\n")
        export_format: TableExportFormat
        for export_format in TableExportFormat:
            fh.write(f"### {export_format}\n\n")
            saving_time_table_path = build_saving_time_table_path(saving_time_table_base_path, export_format)
            saving_time_table = t.cast(pd.DataFrame, pd.read_feather(saving_time_table_path))
            fh.write(saving_time_table.to_markdown())
            fh.write("\n\n")
        fh.write("# Fastest exports\n\n")
        fh.write(fastest_exports.to_markdown())
    logger.info("Benchmark report generated")


def main(argv: t.Union[t.Sequence[str], None]) -> None:
    parser = argparse.ArgumentParser()
    parser.description = "Benchmark the saving of DataFrames in different formats"
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--clear-cache",
        "-c",
        default=False,
        action="store_true",
        help="Clear the cached results of the benchmark before running it",
    )
    args = parser.parse_args(argv)
    # run the benchmark :
    try:
        benchmark(args.config_dir, clear_cache=args.clear_cache)
    except Exception as exc:
        err_msg = f"An error occurred during the benchmark: {exc}"
        logger.error(err_msg)
        raise
    except (KeyboardInterrupt, SystemExit):
        logger.info("The benchmark has been interrupted")
        sys.exit(1)
    except:
        logger.exception("An unexpected error occurred during the benchmark")
        raise


if __name__ == "__main__":
    # e.g.: python src/antares_performance_tests/dataframe_saving_benchmark.py reports/dataframe_saving_benchmark
    main(sys.argv[1:])
