import os
from typing import Any, Dict, Final, List, Optional

import requests

from .database import DatabasePyKx
from .kdbai_exception import KDBAIException
from .utils import process_result
from .version import check_version


os.environ['IGNORE_QHOME'] = '1'
os.environ['PYKX_NOQCE'] = '1'
os.environ['SKIP_UNDERQ'] = '1'
os.environ['QARGS'] = '--unlicensed'
import pykx as kx  # noqa: E402


class SessionPyKx:
    """Session represents a connection to a KDB.AI instance using QIPC connection."""
    def __init__(self,
                 api_key: Optional[str],
                 *,
                 host: Optional[str],
                 port: Optional[int],
                 endpoint: Optional[str],
                 options: Dict[str, Any]):
        """Create a QIPC API connection to a KDB.AI endpoint.

        Args:
            api_key: API Key to be used for authentication.
            endpoint: Server endpoint to connect to.
            host: Server host.
            port: Server port number.
            options: Extra options to create QIPC connection.

        Example:
            Open a session on a custom KDB.AI instance on http://localhost:8082:

            ```python
            session = kdbai.Session(host='localhost' port=8082)
            ```

            Open a session on a custom KDB.AI instance on http://localhost:8082 with authentication:

            ```python
            session = kdbai.Session(host='localhost' port=8082, options={'username': 'John', 'password': 'test'})
            ```
        """
        self.api_key = api_key
        if host is None or port is None:
            if endpoint is None:
                raise KDBAIException('Either host and port or endpoint must be provided')
            else:
                url = requests.utils.urlparse(endpoint)
                host = url.hostname
                port = url.port
        if not host or not port:
            raise KDBAIException(f'Host and port must not be empty: "{host}:{port}"')
        self.host = host
        self.port = port
        try:
            self._gw = kx.SyncQConnection(host=host, port=port, no_ctx=True, **options)
        except Exception as e:
            if len(e.args) and isinstance(e.args[0], str) and \
                (e.args[0] == 'Authentication error' or e.args[0].startswith('access:')):
                raise KDBAIException('Authentication error')
            else:
                raise KDBAIException('Error during creating connection, make sure KDB.AI server is running'
                                    f' and accepts QIPC connection on port {port}: {e}')

        self._is_alive: bool = True
        check_version(self.version())
        self.operator_map: Final[Dict[str, Any]] = self._send_request('getSupportedFilters', None).py()

    def close(self) -> None:
        """Close connection to the server"""
        self._is_alive = False
        self._gw.close()

    def version(self) -> dict:
        """Retrieve version information from server"""
        return {k: v.decode('utf-8') for k, v in self._send_request('getVersion', None).py().items()}

    def create_database(self, database: str) -> DatabasePyKx:
        """Create a new database"""
        result = self._send_request('createDatabase', {'database': database})
        process_result(result)
        return DatabasePyKx(name=database, session=self, tables_meta=dict())

    def databases(self, include_tables: bool) -> List[DatabasePyKx]:
        """List databases"""
        result = self._send_request('listDatabases', None)
        result = process_result(result)
        if include_tables:
            return [self.database(db_name) for db_name in result]

        return [DatabasePyKx(name=db_name, session=self, tables_meta=dict()) for db_name in result]

    def database(self, database: str) -> DatabasePyKx:
        """Fetch a database"""
        result = self._send_request('getDatabase', {'database': database})
        result = process_result(result)
        return DatabasePyKx(name=database, session=self, tables_meta=result.get('tables') or dict())
    
    def databases_info(self) -> Dict[str, Any]:
        """Get Databases Info"""
        result = self._send_request('getAllDatabasesInfo', None)
        info = process_result(result)
        # convert databses and tables output to a row like form instead of column like form
        db=info['databases']
        out = [dict(zip(db.keys(), values)) for values in zip(*db.values())]
        for t in out:
            if t['tables']:
                t['tables']=[dict(zip(t['tables'].keys(), values)) for values in zip(*t['tables'].values())]
            else:
                t['tables'] = []
        info['databases'] = out
        return info

    def session_info(self) -> Dict[str, Any]:
        """Get Session Info"""
        result = self._send_request('getSessionInfo', None)
        info = process_result(result)
        info['sessions'] = [dict(zip(info['sessions'].keys(), values)) for values in zip(*info['sessions'].values())]
        return info
    
    def process_info(self) -> Dict[str, Any]:
        """Get Process Info"""
        result = self._send_request('getProcessInfo', None)
        info = process_result(result)
        info['processes'] = [dict(zip(info['processes'].keys(), values)) for values in zip(*info['processes'].values())]
        return info
    
    def system_info(self) -> Dict[str, Any]:
        """Get System Info"""
        result = self._send_request('getSystemInfo', None)
        info = process_result(result)
        info['sessionsInfo']['sessions'] = [dict(zip(info['sessionsInfo']['sessions'].keys(), values)) for values in zip(*info['sessionsInfo']['sessions'].values())]
        info['processesInfo']['processes'] = [dict(zip(info['processesInfo']['processes'].keys(), values)) for values in zip(*info['processesInfo']['processes'].values())]
        db=info['databasesInfo']['databases']
        out = [dict(zip(db.keys(), values)) for values in zip(*db.values())]
        for t in out:
            if t['tables']:
                t['tables']=[dict(zip(t['tables'].keys(), values)) for values in zip(*t['tables'].values())]
            else:
                t['tables'] = []
        info['databasesInfo']['databases'] = out
        return info

    def _send_request(self, endpoint: str, *args, **kwargs):
        """Send request through QIPC connection"""
        if not self._is_alive:
            raise RuntimeError('Attempted to use closed session')
        try:
            return self._gw(endpoint, *args, **kwargs)
        except RuntimeError:
            raise RuntimeError('Error during request, make sure KDB.AI server running')
