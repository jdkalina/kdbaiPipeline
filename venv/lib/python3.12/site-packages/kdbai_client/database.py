from copy import deepcopy
from typing import Any, Dict, Final, List, Optional, Union

from .constants import _pytype_to_qtype
from .kdbai_exception import KDBAIException
from .table import TablePyKx
from .utils import process_result


class DatabasePyKx:
    """Database implementaion using QIPC connection"""

    def __init__(self, name: str, session, tables_meta: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        self.name: Final[str] = name
        self._session = session
        self._send_request = self._session._send_request
        self._tables: List[TablePyKx] = []
        self.update_tables(tables_meta)

    def create_table(self,
                     table: str,
                     schema: Optional[List[Dict[str, Any]]],
                     indexes: Optional[List[Dict[str, Any]]],
                     partition_column: Optional[str],
                     embedding_configurations: Optional[Dict[str, Any]],
                     external_data_references: Optional[List[Dict[str, Any]]],
                     default_result_type: str) -> TablePyKx:
        """Create new table"""
        if schema is not None:
            schema = deepcopy(schema)
            for column in schema:
                try:
                    column['type'] = _pytype_to_qtype[column['type']]
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

        result = self._send_request('createTable', payload)
        meta = process_result(result)
        table = TablePyKx(name=table,
                          meta=meta,
                          database=self,
                          default_result_type=default_result_type)
        self._tables.append(table)
        return table

    def refresh(self) -> None:
        """Refresh database"""
        result = self._send_request('getDatabase', {'database': self.name})
        result = process_result(result)
        self.update_tables(result.get('tables') or dict())

    def drop(self) -> bool:
        """Delete database"""
        result = self._send_request('deleteDatabase', {'database': self.name})
        process_result(result)
        self._tables = []
        return True

    @property
    def tables(self) -> List[TablePyKx]:
        """Return list of tables in database"""
        return self._tables

    def table(self, table: str) -> TablePyKx:
        """Retrieve a table"""
        result = self._send_request('getTable', {'database': self.name, 'table': table})
        meta = process_result(result)
        for t in self._tables:
            if t.name == table:
                t.update_meta(meta)
                return t

        table = TablePyKx(name=table, meta=meta, database=self)
        self._tables.append(table)
        return table

    def update_tables(self, tables_meta: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Updates tables info when creating/refreshing database"""
        if isinstance(tables_meta, dict):
            table_names = set(tables_meta.get('table', []))
            tables_meta = [{k: v[index] for k, v in tables_meta.items()} for index in range(len(table_names))]
        else:
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
                self._tables.append(TablePyKx(name=meta['table'], meta=meta, database=self))

    def remove_table(self, table_name: str) -> None:
        """Removes table from cached table list, only called from table instance"""
        self._tables = [t for t in self._tables if t.name != table_name]
    
    def info(self) -> Dict[str, Any]:
        """Retrieve database info"""
        result = self._send_request('getDatabaseInfo', {'database': self.name})
        info=process_result(result, result_type='py')
        # convert tables output to a row like form instead of column like form
        tables=info['tables']
        if tables:
            out = [dict(zip(tables.keys(), values)) for values in zip(*tables.values())]
        else:
            out = []
        info['tables']=out
        return info
