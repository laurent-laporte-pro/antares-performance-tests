import argparse
import configparser
import dataclasses
import logging
import logging.config
import sys
import time
import typing as t
from pathlib import Path

import httpx
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("get_studies_benchmark")


def get_access_token(url: str, username: str, password: str) -> str:
    client = httpx.Client(verify=False)
    res = client.post(
        f"{url}/v1/login",
        json={"username": username, "password": password},
    )
    res.raise_for_status()
    credentials = res.json()
    return t.cast(str, credentials["access_token"])


def process_benchmark(
    base_url: str, token: str, nb_iterations: int = 200, timeout: float = 60
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Analyse the duration of the GET `/v1/studies` request.
    """
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        timeout=timeout,
    )
    times = np.zeros(nb_iterations, dtype=np.float64)
    durations = np.zeros(nb_iterations, dtype=np.float64)

    logger.info(f"Starting benchmark with {nb_iterations} iterations...")
    for i in range(nb_iterations):
        start = time.time()
        res = client.get("/v1/studies")
        duration = time.time() - start
        msg = f"=> Iteration {i + 1}/{nb_iterations}: {duration:.3f} s, status code: {res.status_code}"
        if res.status_code == 401:
            logger.warning(f"{msg}, response: {res.json()!r}")
        elif res.status_code == 200:
            logger.info(msg)
        else:
            res.raise_for_status()
        times[i] = start
        durations[i] = duration
    logger.info(f"Finished benchmark with {nb_iterations} iterations")

    # rebase times to start at 0
    times = times - np.min(times)
    return times, durations


def generate_benchmark_report(
    base_url: str,
    durations: npt.NDArray[np.float64],
    output_file: Path,
    *,
    graph_name: str = "graph.png",
    nb_classes: int = 20,
) -> None:
    logger.info(f"Generating benchmark report: '{output_file}'...")
    min_duration = np.min(durations) - 0.001
    max_duration = np.max(durations) + 0.001
    class_width = (max_duration - min_duration) / nb_classes
    with output_file.open(mode="w", encoding="utf-8") as f:
        print("## Analyse de la durée de chargement de la liste des études", file=f)
        print(file=f)
        print(f"- URL de basse de l'API : {base_url}", file=f)
        print(f"- Nombre d'itérations : {durations.size}", file=f)
        print(file=f)
        print(f"![Durée de chargement de la liste des études]({graph_name})", file=f)
        print(file=f)
        print("Résultats statistiques :", file=f)
        print(f"- Durée de chargement minimale : {np.min(durations):.3f} s", file=f)
        print(f"- Durée de chargement moyenne  : {np.mean(durations):.3f} s", file=f)
        print(f"- Durée de chargement maximale : {np.max(durations):.3f} s", file=f)
        print(f"- Écart-type de la durée       : {np.std(durations):.3f} s", file=f)
        print(file=f)
        print("Distribution des temps de chargement :", file=f)
        print(file=f)
        print("|    Classe     | Effectif | Fréquence |", file=f)
        print("|:-------------:|---------:|----------:|", file=f)
        for i in range(nb_classes):
            class_min = min_duration + i * class_width
            class_max = class_min + class_width
            class_duration = [duration for duration in durations if class_min <= duration < class_max]
            class_effectif = len(class_duration)
            class_frequency = class_effectif / durations.size
            print(
                f"| {class_min:.3f} - {class_max:.3f} | {class_effectif:8d} | {class_frequency:9.0%} |",
                file=f,
            )
        print(file=f)
        print("> **Conclusions :**", file=f)
        print("> ", file=f)
        print("> TODO", file=f)
        print(file=f)


def draw_benchmark_report(
    times: npt.NDArray[np.float64],
    durations: npt.NDArray[np.float64],
    output_file: Path,
    width: int = 1024,
    height: int = 768,
) -> None:
    ax = plt.subplots()[1]
    ax.scatter(times, durations, marker="o", color="tomato", s=5)
    ax.set_xlabel("Temps de début de la requête (s)")
    ax.set_ylabel("Durée de la requête (s)")
    ax.set_title("Durée de chargement de la liste des études")
    ax.set_xlim(np.min(times), np.max(times))
    ax.set_ylim(0, np.max(durations) + 0.01)
    ax.grid(True)
    ax.figure.set_size_inches(width / 100, height / 100)
    plt.savefig(output_file, dpi=100)


def analyse_get_studies_duration(
    base_url: str,
    token: str,
    report_file: Path,
    graph_path: Path,
    *,
    nb_iterations: int = 200,
    nb_classes: int = 10,
    timeout: float = 60,
    width: int = 1024,
    height: int = 768,
) -> None:
    times, durations = process_benchmark(
        base_url,
        token,
        nb_iterations=nb_iterations,
        timeout=timeout,
    )
    generate_benchmark_report(
        base_url,
        durations,
        report_file,
        graph_name=graph_path.name,
        nb_classes=nb_classes,
    )
    draw_benchmark_report(
        times,
        durations,
        graph_path,
        width=width,
        height=height,
    )


@dataclasses.dataclass
class BenchmarkConfig:
    nb_iterations: int
    nb_classes: int
    timeout: float

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "BenchmarkConfig":
        return cls(
            nb_iterations=config.getint("benchmark", "nb_iterations"),
            nb_classes=config.getint("benchmark", "nb_classes"),
            timeout=config.getfloat("benchmark", "timeout"),
        )


@dataclasses.dataclass
class GraphConfig:
    width: int
    height: int

    @classmethod
    def from_config(cls, config: configparser.ConfigParser) -> "GraphConfig":
        return cls(
            width=config.getint("graph", "width"),
            height=config.getint("graph", "height"),
        )


@dataclasses.dataclass
class ServerConfig:
    url: str
    username: str
    password: str

    @classmethod
    def from_config(cls, config: configparser.ConfigParser, section: str) -> "ServerConfig":
        return cls(
            url=config.get(section, "url"),
            username=config.get(section, "username"),
            password=config.get(section, "password"),
        )


@dataclasses.dataclass
class ReportConfig:
    report_file: Path
    graph_file: Path

    @classmethod
    def from_config(
        cls,
        config: configparser.ConfigParser,
        config_dir: Path,
    ) -> "ReportConfig":
        report_file = Path(config.get("report", "report_file")).expanduser()
        graph_file = Path(config.get("report", "graph_file")).expanduser()
        if not report_file.is_absolute():
            report_file = config_dir / report_file
        if not graph_file.is_absolute():
            graph_file = config_dir / graph_file
        return cls(report_file=report_file, graph_file=graph_file)


@dataclasses.dataclass
class Config:
    benchmark: BenchmarkConfig
    graph: GraphConfig
    report: ReportConfig
    servers: t.MutableMapping[str, ServerConfig] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_config(cls, config: configparser.ConfigParser, config_dir: Path) -> "Config":
        servers = {}
        for section in config.sections():
            if section.startswith("server."):
                server_name = section.split(".", 1)[1]
                servers[server_name] = ServerConfig.from_config(config, section)
        return cls(
            benchmark=BenchmarkConfig.from_config(config),
            graph=GraphConfig.from_config(config),
            report=ReportConfig.from_config(config, config_dir),
            servers=servers,
        )


# Default `benchmark.ini` configuration file:
BENCHMARK_INI = """[benchmark]
nb_iterations = 50
nb_classes = 10
timeout = 60

[graph]
width = 1024
height = 768

[report]
report_file = get_studies_benchmark.md
graph_file = get_studies_benchmark.png

[server.dev]
url = http://0.0.0.0:8080
username = admin
password = admin

[server.recette]
url = https://antares-web-recette.rte-france.com/api
username = admin
password = ********

[server.prod]
url = https://antares-web.rte-france.com/api
username = admin
password = ********
"""

# Default `logging.ini` configuration file:
LOGGING_INI = """[loggers]
keys = root,get_studies_benchmark

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = DEBUG
handlers = console

[logger_get_studies_benchmark]
level = DEBUG
handlers = console
qualname = get_studies_benchmark
propagate = 0

[handler_console]
class = StreamHandler
level = DEBUG
formatter = generic
args = (sys.stdout,)

[formatter_generic]
format = %(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt =
"""


def benchmark(config_dir: Path) -> None:
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
    for server_name, server_config in config.servers.items():
        logger.info(f"Analysing {server_name}...")
        token = get_access_token(
            server_config.url,
            username=server_config.username,
            password=server_config.password,
        )
        analyse_get_studies_duration(
            server_config.url,
            token,
            config.report.report_file.with_suffix(f".{server_name}.md"),
            config.report.graph_file.with_suffix(f".{server_name}.png"),
            nb_iterations=config.benchmark.nb_iterations,
            nb_classes=config.benchmark.nb_classes,
            timeout=config.benchmark.timeout,
            width=config.graph.width,
            height=config.graph.height,
        )


def main(argv: t.Union[t.Sequence[str], None]) -> None:
    parser = argparse.ArgumentParser()
    parser.description = "Analyse the duration of the GET `/v1/studies` request."
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Path to the configuration directory",
    )
    args = parser.parse_args(argv)
    # run the benchmark :
    try:
        benchmark(args.config_dir)
    except Exception as exc:
        err_msg = f"An error occurred during the benchmark: {exc}"
        logger.error(err_msg)
        sys.exit(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("The benchmark has been interrupted")
        sys.exit(1)
    except:
        logger.exception("An unexpected error occurred during the benchmark")
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
