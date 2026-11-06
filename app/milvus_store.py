"""Snapshot-scoped Milvus adapter. Does not silently fall back to another backend."""
from hashlib import sha256
import json,re
import numpy as np


def collection_name(namespace,snapshot_id):
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,35}',namespace) or not re.fullmatch(r'[a-f0-9]{32}',snapshot_id):
        raise ValueError('Invalid isolated collection identity')
    return namespace+'_'+snapshot_id


def checked_vectors(values,dimension):
    matrix=np.ascontiguousarray(values,dtype=np.float32)
    if matrix.ndim!=2 or matrix.shape[1]!=dimension or not np.isfinite(matrix).all():
        raise ValueError('Invalid vector matrix')
    return matrix


def metadata_rows(snapshot_id,children,vectors):
    required={'child_id','parent_id','document_version_id','file_hash','source','row'}
    if len(children)!=len(vectors):raise ValueError('Vector/Child count mismatch')
    seen=set();records=[]
    for row,(child,vector) in enumerate(zip(children,vectors)):
        if not required<=child.keys() or child['row']!=row or child['child_id'] in seen:
            raise ValueError('Invalid or duplicate Child mapping')
        seen.add(child['child_id'])
        metadata={key:child[key] for key in sorted(required)}
        encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        proof=sha256(encoded+np.asarray(vector,dtype='<f4').tobytes()).hexdigest()
        records.append(dict(id=row,vector=np.asarray(vector,dtype=np.float32).tolist(),snapshot_id=snapshot_id,
            child_id=child['child_id'],parent_id=child['parent_id'],
            document_version_id=child['document_version_id'],file_hash=child['file_hash'],
            metadata=metadata,record_sha256=proof))
    return records


class MilvusSnapshotIndex:
    """Immutable collection handle; row IDs preserve the existing evidence contract.

    L2 is squared Euclidean distance. No normalization or similarity conversion is
    applied. FLAT intentionally preserves exact ordering before ANN evaluation.
    """
    def __init__(self,client,collection,snapshot_id,dimension,count):
        self.client,self.collection,self.snapshot_id=client,collection,snapshot_id
        self.d,self.ntotal=dimension,count

    @classmethod
    def connect(cls,uri,namespace,snapshot_id,dimension,count,token=''):
        from urllib.parse import urlparse
        target=urlparse(uri)
        import os
        permitted=target.scheme=='http' and target.hostname in {'127.0.0.1','localhost'} and target.port==19531
        if os.getenv('RAG_DEPLOYMENT_MODE')=='docker':
            from isolated_config import EXPECTED
            permitted=uri==EXPECTED['milvus_uri']=='http://standalone:19530'
        if not permitted or target.username or target.password or target.query or target.fragment:
            raise ValueError('Unverified Milvus endpoint; no fallback permitted')
        from pymilvus import MilvusClient
        return cls(MilvusClient(uri=uri,token=token,timeout=15),collection_name(namespace,snapshot_id),snapshot_id,dimension,count)

    def stage(self,vectors,children):
        from pymilvus import DataType
        vectors=checked_vectors(vectors,self.d)
        records=metadata_rows(self.snapshot_id,children,vectors)
        if len(records)!=self.ntotal:raise ValueError('Snapshot count mismatch')
        if self.client.has_collection(self.collection):
            raise FileExistsError('Existing collection must be verified, never overwritten')
        schema=self.client.create_schema(auto_id=False,enable_dynamic_field=False)
        schema.add_field('id',DataType.INT64,is_primary=True)
        schema.add_field('vector',DataType.FLOAT_VECTOR,dim=self.d)
        for key,size in [('snapshot_id',32),('child_id',128),('parent_id',128),('document_version_id',128),('file_hash',128),('record_sha256',64)]:
            schema.add_field(key,DataType.VARCHAR,max_length=size)
        schema.add_field('metadata',DataType.JSON)
        indexes=self.client.prepare_index_params();indexes.add_index(field_name='vector',index_type='FLAT',metric_type='L2')
        self.client.create_collection(collection_name=self.collection,schema=schema,index_params=indexes,consistency_level='Strong')
        for offset in range(0,len(records),256):
            self.client.insert(collection_name=self.collection,data=records[offset:offset+256])
        self.client.flush(collection_name=self.collection)
        self.client.load_collection(collection_name=self.collection)
        return self.validate(records)

    def validate(self,expected):
        description=self.client.describe_collection(collection_name=self.collection)
        fields={field['name']:field for field in description['fields']}
        if int(fields['vector']['params']['dim'])!=self.d:raise ValueError('Milvus vector dimension mismatch')
        indexes=self.client.list_indexes(collection_name=self.collection)
        vector_indexes=[self.client.describe_index(collection_name=self.collection,index_name=name) for name in indexes]
        if not any(i.get('field_name')=='vector' and i.get('metric_type')=='L2' and i.get('index_type')=='FLAT' for i in vector_indexes):
            raise ValueError('Milvus distance/index configuration mismatch')
        actual={}
        iterator=self.client.query_iterator(collection_name=self.collection,batch_size=256,filter='',output_fields=['id','snapshot_id','child_id','parent_id','document_version_id','file_hash','metadata','record_sha256','vector'],consistency_level='Strong')
        try:
            while True:
                batch=iterator.next()
                if not batch:break
                for item in batch:
                    if item['id'] in actual:raise ValueError('Duplicate vector row')
                    encoded=json.dumps(item['metadata'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
                    proof=sha256(encoded+np.asarray(item['vector'],dtype='<f4').tobytes()).hexdigest()
                    if item['snapshot_id']!=self.snapshot_id or proof!=item['record_sha256'] or any(item[k]!=item['metadata'][k] for k in ['child_id','parent_id','document_version_id','file_hash']):
                        raise ValueError('Milvus payload integrity mismatch')
                    actual[item['id']]=proof
        finally:iterator.close()
        wanted={item['id']:item['record_sha256'] for item in expected}
        if actual!=wanted or len(actual)!=self.ntotal:raise ValueError('Milvus complete collection mismatch')
        return dict(backend='milvus',collection=self.collection,snapshot_id=self.snapshot_id,count=self.ntotal,dimension=self.d,metric='L2',index_type='FLAT',records_sha256=sha256(json.dumps(sorted(actual.items())).encode()).hexdigest())

    def search(self,vectors,k,metadata_filter=''):
        vectors=checked_vectors(vectors,self.d)
        if not 1<=k<=16384:raise ValueError('Invalid top K')
        distances=np.full((len(vectors),k),np.finfo(np.float32).max,dtype=np.float32);rows=np.full((len(vectors),k),-1,dtype=np.int64)
        if not self.ntotal:return distances,rows
        result=self.client.search(collection_name=self.collection,data=vectors.tolist(),limit=min(k,self.ntotal),filter=metadata_filter,
            search_params={'metric_type':'L2','params':{}},output_fields=['snapshot_id'],consistency_level='Strong')
        if len(result)!=len(vectors):raise ValueError('Missing Milvus result batch')
        for batch,hits in enumerate(result):
            for rank,hit in enumerate(hits):
                row=int(hit['id']);score=float(hit['distance'])
                if not 0<=row<self.ntotal or hit['entity']['snapshot_id']!=self.snapshot_id or not np.isfinite(score):
                    raise ValueError('Cross-snapshot or invalid Milvus result')
                distances[batch,rank]=score;rows[batch,rank]=row
        return distances,rows
