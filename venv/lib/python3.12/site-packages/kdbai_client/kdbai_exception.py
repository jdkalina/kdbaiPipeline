import json
import os

from .constants import CONTENT_TYPE


os.environ['PYKX_IGNORE_QHOME'] = '1'
os.environ['PYKX_NOQCE'] = '1'
os.environ['SKIP_UNDERQ'] = '1'
os.environ['QARGS'] = '--unlicensed'
import pykx as kx #noqa


class KDBAIException(Exception):
    """KDB.AI exception."""

    def __init__(self, msg, e = None, key = None, *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.e = e
        if self.e is not None:
            reason = None
            data = self.e.fp.read()
            try:
                if (self.e.getcode() == 400
                    and self.e.headers.get('Content-type') == CONTENT_TYPE['octet-stream']):
                    reason = kx._wrappers.deserialize(data).py()[0]['ai'].decode('utf-8')
                else:
                    reason = json.loads(data.decode('utf-8'))
                    if key is not None:
                        reason = reason[key]
            except Exception:
                reason = data.decode('utf-8')
            self.code = self.e.code
            if reason is not None:
                self.text = f'{msg[:-1]}, because of: {reason}.'
            self.args = (self.text,)
