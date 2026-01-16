
import json
from typing import Any, Dict, Final, List, Optional

from pandas import DataFrame
import requests

from .constants import _resttype_to_pytype, _validate_dataframe, Headers, RestPath
from .kdbai_exception import KDBAIException
from .rerankers import BaseReranker
from .utils import df_to_qipc, JsonSerializer, process_response, qipc_to_table


class TableRest:
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
        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_LOAD.format(**data)),
                                      headers=self.database._session._build_headers(Headers.ACCEPT_JSON))
        return process_response(response, expected_status_code=200)['result']

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
        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.delete,
                                      self.database._session._build_url(RestPath.TABLE_DROP.format(**data)),
                                      headers=self.database._session._build_headers(dict()))
        process_response(response, expected_status_code=204, get_as_json=False)
        self.database.remove_table(self.name)

    def train(self, payload: DataFrame) -> bool:
        """Trains IVF and IVFPQ indexes"""
        try:
            payload = _validate_dataframe(payload)
        except ValueError as e:
            raise KDBAIException(e.args[0])

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_TRAIN.format(**data)),
                                      data=df_to_qipc(payload),
                                      headers=self.database._session._build_headers(Headers.QIPC_JSON))
        result = process_response(response, expected_status_code=200)['result']
        return result

    def insert(self, payload: DataFrame) -> Dict[str, Any]:
        """Trains IVF and IVFPQ indexes"""
        try:
            payload = _validate_dataframe(payload)
        except ValueError as e:
            raise KDBAIException(e.args[0])

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_INSERT.format(**data)),
                                      data=df_to_qipc(payload),
                                      headers=self.database._session._build_headers(Headers.QIPC_JSON))
        result = process_response(response, expected_status_code=200)['result']
        return result

    def query(self,
              filter: Optional[List[List[Any]]],
              sort_columns: Optional[List[str]],
              group_by: Optional[List[str]],
              aggs: Optional[Dict[str, Any]],
              limit: Optional[int],
              result_type: Optional[str]):
        """Execute similarity search"""
        payload = dict()
        optionals = {
            'sortColumns': sort_columns,
            'groupBy': group_by,
            'aggs': aggs,
            'limit': limit,
            'filter': filter
        }
        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_QUERY.format(**data)),
                                      data=json.dumps(payload, cls=JsonSerializer, default=str),
                                      headers=self.database._session._build_headers(Headers.JSON_QIPC))
        result = process_response(response, expected_status_code=200, get_as_json=False)
        return qipc_to_table(result.content, result_type or self.default_result_type)

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
        """Todo"""
        payload = {
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
            'aggs': aggs,
            'filter': filter
        }

        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_SEARCH.format(**data)),
                                      data=json.dumps(payload, cls=JsonSerializer),
                                      headers=self.database._session._build_headers(Headers.JSON_QIPC))
        result = process_response(response, expected_status_code=200, get_as_json=False)
        return qipc_to_table(result.content, result_type or self.default_result_type, is_list=True)

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
            'indexes': indexes
        }
        if parts is not None:
            payload['parts'] = parts

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.INDEX_UPDATE.format(**data)),
                                      data=json.dumps(payload, cls=JsonSerializer),
                                      headers=self.database._session._build_headers(Headers.JSON_JSON))
        result = process_response(response, expected_status_code=202, get_as_json=False)
        return result

    def delete_data(self,
              filter: Optional[List[List[Any]]],
              result_type: Optional[str]):
        """Execute delete data operation on a table"""
        payload = dict()
        optionals = {
            'filter': filter
        }
        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_DELETE_DATA.format(**data)),
                                      data=json.dumps(payload, cls=JsonSerializer, default=str),
                                      headers=self.database._session._build_headers(Headers.JSON_JSON))
        return process_response(response, expected_status_code=200)['result']
    
    def update_data(self,
              filter: Optional[List[List[Any]]],
              columns: dict,
              result_type: Optional[str]):
        """Execute update data operation on a table"""
        payload = {
            'columns': columns
        }
        optionals = {
            'filter': filter
        }
        for k, v in optionals.items():
            if v is not None:
                payload[k] = v

        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.post,
                                      self.database._session._build_url(RestPath.TABLE_UPDATE_DATA.format(**data)),
                                      data=json.dumps(payload, cls=JsonSerializer, default=str),
                                      headers=self.database._session._build_headers(Headers.JSON_JSON))
        return process_response(response, expected_status_code=200)['result']

    @property
    def schema(self):
        """Table schema"""
        return self._meta.get('schema', [])

    @property
    def indexes(self) -> List[Dict[str, Any]]:
        """Return list of indexes defined on the table"""
        return self._meta.get('index', [])

    def update_meta(self, meta: dict) -> None:
        """Updates metadata"""
        self._meta = meta
        schema = self._meta.get('schema', None)
        if schema is not None:
            for column in schema:
                col_type = column['type']
                column['type'] = 'general' if col_type in ['', ' '] else _resttype_to_pytype[col_type]
                if not column['attributes']:
                    column.pop('attributes', None)
    
    def info(self) -> Dict[str, Any]:
        """Retrieve table info"""
        data = {'db_name': self.database.name, 'table_name': self.name}
        response = self._send_request(requests.get,
                                      self.database._session._build_url(RestPath.TABLE_INFO_GET.format(**data)))
        return process_response(response, expected_status_code=200)['result']