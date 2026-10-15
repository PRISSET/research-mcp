import logging

from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
