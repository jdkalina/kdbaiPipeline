from copy import deepcopy
from typing import Any, Dict, Final, List, Optional

from pandas import DataFrame, Series

from .constants import _qtype_to_pytype, _validate_dataframe
from .kdbai_exception import KDBAIException
from .rerankers import BaseReranker
from .utils import process_result


def _convert_filter(operator_map:dict, filters: list) -> list:
    """Converts filters to appropriate format for the server

    Args:
        operator_map (dict): Dictionary that contains pykx value for filter operators.
        filters (list): list of filter conditions.

    Returns:
        List of conditions that is accepted by the server
    """
    def _convert_filter_inner(filters: list) -> list:
        if isinstance(filters[0], (list, tuple)):
            # this is for the operands of `and`, `or` and top level conditions
            return [_convert_filter_inner(list(x)) for x in filters]
        operator = filters[0]
        if operator in ['or', 'and', 'not']:
            return [operator_map.get(operator, operator), *[_convert_filter_inner(x) for x in filters[1:]]]
        if operator == '>=':
            return _convert_filter_inner(['not', ['<', *filters[1:]]])
        if operator == '<=':
            return _convert_filter_inner(['not', ['>', *filters[1:]]])
        if operator == '<>':
            return _convert_filter_inner(['not', ['=', *filters[1:]]])
        if operator == 'like' and isinstance(filters[2], str):
            pattern = filters[2].encode()
            if len(pattern) == 1:
                pattern = [pattern]
            filters[2] = pattern
        try:
            # `in` needs enlist ("in", "colname", [['AA', 'BB']])
            if operator == 'in' and isinstance(filters[2][0], str):
                filters[2] = [filters[2]]
        except KeyError:
            # we can't figure out, server will respond if parameter is incorrect
            pass
        if operator not in operator_map:
            raise KDBAIException(f"Unsupported filter function: {operator}")
        return [operator_map.get(operator, operator), *filters[1:]]

    # we make changes inplace and don't want any side effect for the user's data
    filters = deepcopy(filters)
    if isinstance(filters, tuple):
        filters = list(filters)
    return _convert_filter_inner(filters)


class TablePyKx:
    """KDB.AI table."""

    def __init__(self,
                 name: str,
                 meta: dict,
                 database,
                 default_result_type: str = 'pd'
                 ) -> None:

        self.name: Final[str] = name
        self._meta: Optional[dict] = None
        self.update_meta(meta)
        self.database: Final = database
        self._send_request: Final = self.database._send_request
        self.default_result_type: str = default_result_type

    def refresh(self) -> None:
        """Refresh meta for table"""
        self.database.table(self.name)

    def load(self) -> None:
        """Loads external table"""
        result = self._send_request('loadTable', {'database': self.database.name, 'table': self.name})
        return process_result(result, result_type='py')

    def index(self, name: str) -> Dict[str, Any]:
        """Return specific index"""
        # uses locally cached indexes, if index delete/create/update is implemented, this
        # - has to always fetch index from server
        # - method can get refresh bool parameter
        for index in self.indexes:
            if index['name'] == name:
                return index
        raise KDBAIException('index not present')

    def drop(self) -> None:
        """Drops table"""
        result = self._send_request('deleteTable', {'database': self.database.name, 'table': self.name})
        process_result(result)
        self.database.remove_table(self.name)

    def train(self, payload: DataFrame) -> bool:
        """Trains IVF and IVFPQ indexes"""
        try:
            payload = _validate_dataframe(payload)
        except ValueError as e:
            raise KDBAIException(e.args[0])
        result = self._send_request('trainData', {'database': self.database.name,
                                                  'table': self.name,
                                                  'payload': payload})
        return process_result(result, result_type='py')

    def insert(self, payload: DataFrame) -> Dict[str, Any]:
        """Insert data into table"""
        try:
            payload = _validate_dataframe(payload)
        except ValueError as e:
            raise KDBAIException(e.args[0])
        result = self._send_request('insertData', {'database': self.database.name,
                                                   'table': self.name,
                                                   'payload': payload})
        return process_result(result, result_type='py')

    def query(self,
              filter: Optional[List[List[Any]]],
              sort_columns: Optional[List[str]],
              group_by: Optional[List[str]],
              aggs: Optional[Dict[str, Any]],
              limit: Optional[int],
              result_type: Optional[str]):
        """Todo"""
        payload = {
            'database': self.database.name,
            'table': self.name
        }
        optionals = {
            'sortColumns': sort_columns,
            'groupBy': group_by,
            'aggs': aggs,
            'limit': limit
        }
        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        if filter:
            payload['filter'] = _convert_filter(self.database._session.operator_map, filter)

        result = self._send_request('query', payload)
        return process_result(result, result_type=result_type or self.default_result_type)

    def search(self,
               vectors: Dict[str, Any],
               n: Optional[int],
               *,
               range: Optional[float],
               type: Optional[str],
               index_params: Optional[Dict[str, Any]],
               options: Optional[Dict[str, Any]],
               filter: Optional[List[List[Any]]],
               sort_columns: Optional[List[str]],
               group_by: Optional[List[str]],
               search_by: Optional[List[str]],
               aggs: Optional[Dict[str, Any]],
               result_type: Optional[str]):
        """Execute similarity search"""
        vectors = {k: v.tolist() if isinstance(v, Series) else v for k, v in vectors.items()}
        payload = {
            'database': self.database.name,
            'table': self.name,
            'vectors': vectors,
        }
        optionals = {
            'n': n,
            'range': float(range) if range is not None else range,
            'type': type,
            'indexParams': index_params,
            'options': options,
            'sortColumns': sort_columns,
            'groupBy': group_by,
            'searchBy': search_by,
            'aggs': aggs
        }

        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        if filter:
            payload['filter'] = _convert_filter(self.database._session.operator_map, filter)

        result = self._send_request('search', payload)
        return process_result(result, result_type=result_type or self.default_result_type, is_list=True)

    def search_and_rerank(self,
                        vectors: Dict[str, Any],
                        n: int,
                        *,
                        reranker: BaseReranker,
                        queries: List[str],
                        text_column: str,
                        type: Optional[str],
                        index_params: Optional[Dict[str, Any]],
                        options: Optional[Dict[str, Any]],
                        filter: Optional[List[List[Any]]],
                        result_type: Optional[str]):

        new_n = n*reranker.overfetch_factor

        result = self.search(vectors, new_n, range=None, type=type, 
                             index_params=index_params,
                             options=options, filter=filter, 
                             sort_columns=None,
                             group_by=None, 
                             search_by=None,
                             aggs=None,
                             result_type=result_type)

        return [reranker.rank(query=q, results=r, text_column=text_column, top_n=n) for q, r in zip(queries, result)]

    def update_indexes(self, indexes: List[str], parts: Optional[List[Any]]) -> bool:
        """Updates indexes"""
        payload = {
            'database': self.database.name,
            'table': self.name,
            'indexes': indexes
        }
        if parts is not None:
            payload['parts'] = parts

        result = self._send_request('updateIndexes', payload)
        return process_result(result)
    
    def delete_data(self,
              filter: Optional[List[List[Any]]],
              result_type: Optional[str]):
        """delete data from table"""
        payload = {
            'database': self.database.name,
            'table': self.name
        }
  
        if filter:
            payload['filter'] = _convert_filter(self.database._session.operator_map, filter)

        result = self._send_request('deleteData', payload)
        return process_result(result, result_type=result_type or self.default_result_type)
    
    def update_data(self,
              filter: Optional[List[List[Any]]],
              columns: dict,
              result_type: Optional[str]):
        """update data from table"""
        payload = {
            'database': self.database.name,
            'table': self.name,
            'columns': columns
        }
 
        if filter:
            payload['filter'] = _convert_filter(self.database._session.operator_map, filter)

        result = self._send_request('updateData', payload)
        return process_result(result, result_type=result_type or self.default_result_type)

    @property
    def schema(self):
        """Table schema"""
        schema = self._meta.get('schema', dict(type=''.encode('utf-8'), name=[]))
        col_types = schema['type'].decode('utf-8')
        col_types = ['' if t == ' ' else t for t in col_types]  # `dict` type is returned as space character
        return [{'name': n, 'type': _qtype_to_pytype[t]} for n, t in zip(schema['name'], col_types)]

    @property
    def indexes(self) -> List[Dict[str, Any]]:
        """Return list of indexes defined on the table"""
        index_data = self._meta.get('index', {})
        if not index_data:
            return []
        index_count = len(index_data.get('name', []))
        return [{k: v[i] for k, v in index_data.items()} for i in range(index_count)]

    def update_meta(self, meta: dict) -> None:
        """Updates metadata"""
        self._meta = meta
    
    def info(self) -> Dict[str, Any]:
        """Get table info"""
        result = self._send_request('getTableInfo', {'database': self.database.name,
                                                   'table': self.name})
        return process_result(result, result_type='py')