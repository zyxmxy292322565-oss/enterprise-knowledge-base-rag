"""Only namespaced Redis operations are exposed to application code."""
class NamespacedRedis:
 def __init__(self,raw,prefix,identity):self._raw=raw;self._prefix=prefix;self._identity=identity
 def _key(self,k):
  if not isinstance(k,str) or not k:raise ValueError('Explicit string key required')
  return self._prefix+k
 def get(self,key):return self._identity if key=='rag:v2:identity' else self._raw.get(self._key(key))
 def set(self,name,value,*args,**kwargs):return self._raw.set(self._key(name),value,*args,**kwargs)
 def setex(self,key,time,value):return self._raw.setex(self._key(key),time,value)
 def incrby(self,key,amount=1):return self._raw.incrby(self._key(key),amount)
 def exists(self,*keys):return self._raw.exists(*(self._key(k) for k in keys))
 def delete(self,*keys):return self._raw.delete(*(self._key(k) for k in keys))
 def expire(self,key,time):return self._raw.expire(self._key(key),time)
 def ttl(self,key):return self._raw.ttl(self._key(key))
 def mget(self,keys):return self._raw.mget([self._key(k) for k in keys])
 def mset(self,values):return self._raw.mset({self._key(k):v for k,v in values.items()})
 def scan_iter(self,match=None,**kwargs):
  if any(k not in ['count','_type'] for k in kwargs):raise ValueError('Unsupported scan options')
  for key in self._raw.scan_iter(match=self._prefix+(match or '*'),**kwargs):
   if not key.startswith(self._prefix):raise RuntimeError('Foreign Redis key')
   yield key[len(self._prefix):]
 def ping(self):return self._raw.ping()
 def close(self):return self._raw.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
 def __getattr__(self,name):raise RuntimeError('Redis operation not reviewed for isolation: '+name)
