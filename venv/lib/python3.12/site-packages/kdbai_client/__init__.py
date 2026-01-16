"""KDB.AI Client for Python."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .api import Database, Session, Table  # noqa
from .constants import MAX_DATETIME, MIN_DATETIME  # noqa
from .kdbai_exception import KDBAIException  # noqa
from .version import _set_version  # noqa
from .rerankers import BaseReranker, CohereReranker, JinaAIReranker, VoyageAIReranker


try:
    __version__ = _version('kdbai_client')
    if "dev" in __version__ or __version__ == "0.0.0":
        __version__ = 'dev'
except PackageNotFoundError:  # pragma: no cover
    __version__ = 'dev'
_set_version(__version__)

__all__ = sorted(['__version__', 'KDBAIException', 'MIN_DATETIME', 'MAX_DATETIME', 'Database', 'Session', 'Table'])


def __dir__():
    return __all__
