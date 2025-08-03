"""Collection of utilities for the LMmarkdown project"""

# default logger initialized from here
from .logging import ILogger
from .logging import ConsoleLogger

logger: ILogger = ConsoleLogger()
