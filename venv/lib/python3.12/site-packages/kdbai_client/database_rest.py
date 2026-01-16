from copy import deepcopy
import json
from typing import Any, Dict, Final, List, Optional

import requests

from .constants import _pytype_to_resttype, Headers, RestPath
from .kdbai_exception import KDBAIException
from .table_rest import TableRest
from .utils import JsonSerializer, process_response


class DatabaseRest:
    """todo"""

    def __init__(self, name: str, session, tables_meta: List[Dict[str, Any]]) -> None:
        self.name: Final[str] = name
        self._session = session
        self._send_request = self._session._send_request
        self._tables: List[TableRest] = []
        self.update_tables(tables_meta)

    def create_table(self,
                     table: str,
                     schema: Optional[List[Dict[str, Any]]],
                     indexes: List[Dict[str, Any]],
                     partition_column: Optional[str],
                     embedding_configurations: Optional[Dict[str, Any]],
                     external_data_references: Optional[List[Dict[str, Any]]],
                     default_result_type: str) -> TableRest:
        """Create new table"""
        if schema is not None:
            schema = deepcopy(schema)
            for column in schema:
                try:
                    column['type'] = _pytype_to_resttype[column['type']]
                except KeyError:
                    raise KDBAIException(f'Invalid column type: {column["type"]}')

        payload = {
            'database': self.name,
            'table': table
        }

        optionals = {
            'schema': schema,
            'indexes': indexes,
            'partitionColumn': partition_column,
            'embeddingConfigurations': embedding_configurations,
            'externalDataReferences': external_data_references
        }
        for key, value in optionals.items():
            if value is not None:
                payload[key] = value

        response = self._send_request(requests.post,
                                      self._session._build_url(RestPath.TABLE_CREATE.format(db_name=self.name)),
                                      data=json.dumps(payload, cls=JsonSerializer),
                                      headers=self._session._build_headers(Headers.JSON_JSON))
        result = process_response(response, expected_status_code=201)

        table = TableRest(name=table,
                          meta=result['result'],
                          database=self,
                          default_result_type=default_result_type)
        self._tables.append(table)
        return table

    def refresh(self) -> None:
        """Refresh database"""
        response = self._send_request(requests.get,
                                      self._session._build_url(RestPath.DATABASE_GET.format(db_name=self.name)),
                                      headers=self._session._build_headers(Headers.ACCEPT_JSON))
        result = process_response(response, expected_status_code=200)['result']
        self.update_tables(result.get('tables') or dict())

    def drop(self) -> bool:
        """Delete database"""
        response = self._send_request(requests.delete,
                                      self._session._build_url(RestPath.DATABASE_DROP.format(db_name=self.name)),
                                      headers=self._session._build_headers(Headers.JSON_JSON))
        process_response(response, expected_status_code=204, get_as_json=False)
        self._tables = []
        return True

    @property
    def tables(self) -> List[TableRest]:
        """Return list of tables in database"""
        return self._tables

    def table(self, table: str) -> TableRest:
        """Retrieve a table"""
        data = {'db_name': self.name, 'table_name': table}
        response = self._send_request(requests.get,
                                      self._session._build_url(RestPath.TABLE_GET.format(**data)),
                                      headers=self._session._build_headers(Headers.JSON_JSON))
        meta = process_response(response, expected_status_code=200)['result']

        for t in self._tables:
            if t.name == table:
                t.update_meta(meta)
                return t

        table = TableRest(name=table, meta=meta, database=self)
        self._tables.append(table)
        return table

    def update_tables(self, tables_meta: List[Dict[str, Any]]) -> None:
        """Updates tables info when creating/refreshing database"""
        table_names = {meta['table'] for meta in tables_meta}
        self._tables = [t for t in self._tables if t.name in table_names]
        prev_table_names = {t.name for t in self._tables}

        for meta in tables_meta:
            table_name = meta['table']
            if table_name in prev_table_names:
                for table in self._tables:
                    if table.name == table_name:
                        table.update_meta(meta)
            else:
                self._tables.append(TableRest(name=meta['table'], meta=meta, database=self))

    def remove_table(self, table_name: str) -> None:
        """Removes table from cached table list, only called from table instance"""
        self._tables = [t for t in self._tables if t.name != table_name]
    
    def info(self) -> Dict[str, Any]:
        """Retrieve database info"""
        data = {'db_name': self.name}
        response = self._send_request(requests.get,
                                      self._session._build_url(RestPath.DATABASE_INFO_GET.format(**data)),
                                      headers=self._session._build_headers(Headers.JSON_JSON))
        return process_response(response, expected_status_code=200)['result']
