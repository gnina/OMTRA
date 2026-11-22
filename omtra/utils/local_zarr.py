"""Read existing local Zarr-v3 arrays without an asyncio/threaded store.

Limited deliberately to this project's unsharded, first-axis chunked bytes/Zstd
and vlen-UTF8 stores. Unsupported encodings fail explicitly. No write API.
"""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from numcodecs import Zstd, VLenUTF8


class LocalArray:
    def __init__(self,path,cache_size=4):
        self.path=Path(path);self.metadata=json.loads((self.path/'zarr.json').read_text())
        m=self.metadata
        if m.get('zarr_format')!=3 or m.get('node_type')!='array':raise ValueError('Expected a Zarr-v3 array')
        if m['chunk_grid']['name']!='regular' or m['chunk_key_encoding']['name']!='default':raise ValueError('Unsupported Zarr chunk layout')
        self.shape=tuple(m['shape']);self.chunks=tuple(m['chunk_grid']['configuration']['chunk_shape'])
        if self.chunks[1:]!=self.shape[1:]:raise ValueError('Only first-axis chunking is supported')
        self.separator=m['chunk_key_encoding']['configuration']['separator']
        self.codecs=m['codecs']
        if any(c['name'] not in ('bytes','zstd','vlen-utf8') for c in self.codecs):raise ValueError('Unsupported Zarr codec')
        self.dtype=np.dtype(object if m['data_type']=='string' else m['data_type'])
        self.cache=OrderedDict();self.cache_size=cache_size

    def chunk(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        coordinates=[index]+[0]*(len(self.shape)-1)
        key='c'+self.separator+self.separator.join(map(str,coordinates))
        payload=(self.path/key).read_bytes()  # Missing chunks fail rather than conceal absent training data.
        for codec in reversed(self.codecs):
            name=codec['name']
            if name=='zstd':payload=Zstd().decode(payload)
            elif name=='vlen-utf8':payload=VLenUTF8().decode(payload)
            elif name=='bytes':
                endian='<' if codec.get('configuration',{}).get('endian','little')=='little' else '>'
                payload=np.frombuffer(payload,dtype=self.dtype.newbyteorder(endian))
        result=np.asarray(payload).reshape(self.chunks)
        self.cache[index]=result
        if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return result

    def read(self,start,end):
        start,end=int(start),int(end)
        if not 0<=start<=end<=self.shape[0]:raise IndexError('Array slice outside stored support')
        if start==end:return np.empty((0,*self.shape[1:]),dtype=self.dtype)
        first,last=start//self.chunks[0],(end-1)//self.chunks[0]
        if first==last:return self.chunk(first)[start%self.chunks[0]:start%self.chunks[0]+end-start]
        parts=[self.chunk(i)[max(0,start-i*self.chunks[0]):min(self.chunks[0],end-i*self.chunks[0])] for i in range(first,last+1)]
        return np.concatenate(parts)
