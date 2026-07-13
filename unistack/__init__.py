import logging

from unistack.core import RunResult, UniStack, UniStackError

# Library convention: emit through logging, never print; the consuming app decides handlers.
logging.getLogger("unistack").addHandler(logging.NullHandler())

__all__ = ["UniStack", "RunResult", "UniStackError"]
