# pyright: reportUnusedImport=false
# flake8: noqa

# default logger initialized from here
from .logging import ILogger
from .logging import ConsoleLogger

logger: ILogger = ConsoleLogger()
