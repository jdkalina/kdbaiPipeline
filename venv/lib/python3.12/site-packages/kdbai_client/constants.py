from datetime import datetime

from numpy import arange
from pandas import DataFrame


# REST related constants
CONTENT_TYPE = {
    'json': 'application/json',
    'octet-stream': 'application/octet-stream'
}
# Minimum datetime/timestmp supported by q/kdb+
MIN_DATETIME = datetime(1707, 9, 22, 0, 12, 43, 145224)
# Maximum datetime/timestamp supported by q/kdb+
MAX_DATETIME = datetime(2262, 4, 11, 23, 47, 16, 854775)

class RestPath:
    """Enum for REST paths"""

    READY           = '/api/v2/ready'
    VERSION         = '/api/v2/version'

    DATABASE_CREATE = '/api/v2/databases'
    DATABASE_LIST   = '/api/v2/databases'
    DATABASE_GET    = '/api/v2/databases/{db_name}'
    DATABASE_DROP   = '/api/v2/databases/{db_name}'

    TABLE_CREATE    = '/api/v2/databases/{db_name}/tables'
    TABLE_DROP      = '/api/v2/databases/{db_name}/tables/{table_name}'
    TABLE_GET       = '/api/v2/databases/{db_name}/tables/{table_name}'
    TABLE_TRAIN     = '/api/v2/databases/{db_name}/tables/{table_name}/train'
    TABLE_INSERT    = '/api/v2/databases/{db_name}/tables/{table_name}/insert'
    TABLE_QUERY     = '/api/v2/databases/{db_name}/tables/{table_name}/query'
    TABLE_SEARCH    = '/api/v2/databases/{db_name}/tables/{table_name}/search'
    TABLE_LOAD      = '/api/v2/databases/{db_name}/tables/{table_name}/load'
    TABLE_DELETE_DATA = '/api/v2/databases/{db_name}/tables/{table_name}/data/delete'
    TABLE_UPDATE_DATA = '/api/v2/databases/{db_name}/tables/{table_name}/data/update'

    INDEX_UPDATE    = '/api/v2/databases/{db_name}/tables/{table_name}/indexes/update'
    
    #Info api
    DATABASE_INFO_GET = '/api/v2/info/databases/{db_name}'
    TABLE_INFO_GET = '/api/v2/info/databases/{db_name}/tables/{table_name}'
    DATABASES_INFO_GET = '/api/v2/info/databases'
    SESSION_INFO_GET = '/api/v2/info/session'
    PROCESS_INFO_GET = '/api/v2/info/process'
    SYSTEM_INFO_GET = '/api/v2/info/system'


class Headers:
    """Predefined headers for REST requests"""

    JSON_JSON = {'Content-Type': CONTENT_TYPE['json'],
                 'Accept': CONTENT_TYPE['json']}
    JSON_QIPC = {'Content-Type': CONTENT_TYPE['json'],
                 'Accept': CONTENT_TYPE['octet-stream']}
    QIPC_JSON = {'Content-Type': CONTENT_TYPE['octet-stream'],
                 'Accept': CONTENT_TYPE['json']}
    ACCEPT_JSON = {'Accept': CONTENT_TYPE['json']}

# new API mapping
_qtype_to_pytype = dict(
    b='bool',
    B='bools',
    g='guid',
    G='guids',
    x='uint8',
    X='uint8s',
    h='int16',
    H='int16s',
    i='int32',
    I='int32s',
    j='int64',
    J='int64s',
    e='float32',
    E='float32s',
    f='float64',
    F='float64s',
    c='char',
    C='bytes',
    s='str',
    S='strs',
    p='datetime64[ns]',
    P='datetime64[ns]s',
    m='datetime64[M]',
    M='datetime64[M]s',
    d='datetime64[D]',
    D='datetime64[D]s',
    n='timedelta64[ns]',
    N='timedelta64[ns]s',
    u='timedelta64[m]',
    U='timedelta64[m]s',
    v='timedelta64[s]',
    V='timedelta64[s]s',
    t='timedelta64[ms]',
    T='timedelta64[ms]s',
)

_qtype_to_pytype[''] = "general"

_pytype_to_qtype = {v: k for k, v in _qtype_to_pytype.items()}

_resttype_to_pytype = dict(
    dict='general',
    boolean='bool',
    booleans='bools',
    guid='guid',
    guids='guids',
    byte='uint8',
    bytes='uint8s',
    short='int16',
    shorts='int16s',
    int='int32',
    ints='int32s',
    long='int64',
    longs='int64s',
    real='float32',
    reals='float32s',
    float='float64',
    floats='float64s',
    char='char',
    chars='bytes',
    symbol='str',
    symbols='strs',
    timestamp='datetime64[ns]',
    timestamps='datetime64[ns]s',
    month='datetime64[M]',
    months='datetime64[M]s',
    date='datetime64[D]',
    dates='datetime64[D]s',
    timespan='timedelta64[ns]',
    timespans='timedelta64[ns]s',
    minute='timedelta64[m]',
    minutes='timedelta64[m]s',
    second='timedelta64[s]',
    seconds='timedelta64[s]s',
    time='timedelta64[ms]',
    times='timedelta64[ms]s',
)

_pytype_to_resttype = {v: k for k, v in _resttype_to_pytype.items()}

def _validate_dataframe(data: DataFrame):
    if data is None or len(data) == 0:
        raise ValueError('dataframe should not be empty.')
    if isinstance(data, dict):
        data = DataFrame(data=data)
    elif isinstance(data, DataFrame) and (data.index.values != arange(len(data))).any():
        data = data.reset_index(drop=True)
    return data
