from datetime import date, datetime
from json import JSONEncoder
import os
from typing import Dict, Final, Union

import numpy as np
from pandas import Series
from requests import Response

from .kdbai_exception import KDBAIException


os.environ['PYKX_IGNORE_QHOME'] = '1'
os.environ['PYKX_NOQCE'] = '1'
os.environ['SKIP_UNDERQ'] = '1'
os.environ['QARGS'] = '--unlicensed'
import pykx as kx  # noqa: E402


conversions: Final[Dict[str, Union[kx.Table, kx.List]]] = {
    'pd': lambda x: x.pd(),
    'py': lambda x: x.py(),
    'q': lambda x: x
}


def _unlicensed_getitem_dict(d: kx.Dictionary, key: str):
    return d._values._unlicensed_getitem(d._keys.py().index(key))

def process_result(result: kx.Dictionary, result_type: str = 'py', is_list: bool = False):
    if not result_type in ["pd", "py", "q"]:
        raise KDBAIException("Unsupported result type. Supported result types: pd, py, q")
    """Effectively processes results from server without unnecessary conversions"""
    if _unlicensed_getitem_dict(result, 'success').py():
        actual_result = _unlicensed_getitem_dict(result, 'result')
        if is_list:
            return [conversions[result_type](item) for item in actual_result]
        else:
            return conversions[result_type](actual_result)
    else:
        raise KDBAIException(_unlicensed_getitem_dict(result, 'error').py().decode('utf-8'))


def process_response(response: Response, expected_status_code: int, get_as_json: bool = True):
    """Check REST response status code is expected, raise raise exception if not"""
    if response.status_code == expected_status_code:
        if get_as_json:
            return response.json()
        return response
    else:
        json = response.json()
        try:
            raise KDBAIException(json['error'])
        except KeyError:
            raise KDBAIException(json)


MAX_QIPC_SERIALIZATION_SIZE = 10*1024*1024  # 10MB

def df_to_qipc(dataframe):
    """Converts dataframe to bystream for effective REST requests"""
    data = bytes(kx._wrappers.k_pickle(kx.toq(dataframe)))
    if len(data) > MAX_QIPC_SERIALIZATION_SIZE:
        raise KDBAIException(
            f'The maximum serialized size of the data to insert is {MAX_QIPC_SERIALIZATION_SIZE} bytes. '
            f'The size of your serialized data is {len(data)} bytes. '
            'Please insert your data by smaller batches.'
            )
    return data


def qipc_to_table(binary_content, result_type: str, is_list: bool = False):
    """Deserialize octet stream result to requested type"""
    result = kx._wrappers.deserialize(binary_content)
    actual_result = _unlicensed_getitem_dict(result, 'result')
    if is_list:
        return [conversions[result_type](item) for item in actual_result]
    return conversions[result_type](actual_result)


class JsonSerializer(JSONEncoder):
    """Convert numpy datatypes to serializable types"""
    def default(self, obj):
        """Convert function"""
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Series):
            return obj.tolist()
        if isinstance(obj, (date, datetime)):
            return str(obj)
        return JSONEncoder.default(self, obj)
