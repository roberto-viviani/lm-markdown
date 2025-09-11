# pyright: reportUnusedImport=false
# flake8: noqa

# default logger initialized from here
from .logging import LoggerBase
from .logging import ConsoleLogger

logger: LoggerBase = ConsoleLogger()
