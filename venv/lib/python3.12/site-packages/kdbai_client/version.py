import logging
from typing import Any, Dict

from packaging import version

from .kdbai_exception import KDBAIException


__version__ = None

def _set_version(version):
    global __version__
    __version__ = version

def get_version():
    """Return version"""
    return __version__

def check_version(versions: Dict[str, Any]) -> None:
    """Check version compatibility between server and client"""
    if __version__ == 'dev':
        logging.warning(
            'You are running a development version of `kdbai_client`.\n'
            'Compatibility with the KDB.AI server is not guaranteed.'
        )
        return

    if (version.parse(__version__) < version.parse(versions['clientMinVersion'])
        or (versions['clientMaxVersion'] != 'latest'
            and version.parse(__version__) > version.parse(versions['clientMaxVersion']))):
        raise KDBAIException(
            f'Your KDB.AI server is not compatible with this client (kdbai_client=={__version__}).\n'
            f"Please use kdbai_client >={versions['clientMinVersion']} and <={versions['clientMaxVersion']}."
        )

    try:
        if version.parse(versions['serverVersion']) < version.Version('1.6.0'):
            raise KDBAIException(
                f'Your KDB.AI server is not compatible with this client (kdbai_client=={__version__}).\n'
                f"Please use kdbai_client <= 1.6.0 or update server to >= 1.6.0."
            )
    except version.InvalidVersion:
        # developer version
        pass
