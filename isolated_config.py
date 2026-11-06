"""Exact local deployment identity and process write barriers; no legacy fallback."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
_CONTAINER=os.environ.get('RAG_DEPLOYMENT_MODE')=='docker'
if _CONTAINER:
 if os.name=='nt':raise RuntimeError('Container contract requires Linux')
 from container_identity import load_contract
 EXPECTED,_CONTAINER_PRIVATE=load_contract('/run/secrets/identity.json','/run/secrets/environment.json')
 CONFIG_PATH=Path('/run/secrets/identity.json')
else:
 CONFIG_PATH=ROOT/'isolated.json'
 EXPECTED=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
def _load():
 if _CONTAINER:
  for key,value in _CONTAINER_PRIVATE.items():
   if key.startswith(('MYSQL_','REDIS_')) and os.environ.get(key) not in [None,'',str(value)]:raise RuntimeError('Unexpected container override: '+key)
   os.environ[key]=str(value)
  return dict(_CONTAINER_PRIVATE)
 script="$s=[Console]::In.ReadToEnd();$v=ConvertTo-SecureString $s;[System.Net.NetworkCredential]::new('',$v).Password"
 raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',script],input=(ROOT/'.isolation/environment.dpapi').read_bytes().strip(),env={k:v for k,v in os.environ.items() if k.upper()!='PSMODULEPATH'},stderr=subprocess.PIPE)
 config=json.loads(raw)
 from dotenv import load_dotenv
 load_dotenv(ROOT/'.env',override=False)
 for key,value in config.items():
  current=os.environ.get(key)
  if key.startswith(('MYSQL_','REDIS_')) or key in ['RAG_V2_MYSQL_UUID','RAG_V2_REDIS_PREFIX','RAG_V2_REDIS_IDENTITY']:
   if current not in [None,'','DPAPI_MANAGED',str(value)]:raise RuntimeError('Unexpected environment override: '+key)
  os.environ[key]=str(value)
 return config
_PRIVATE=_load()
_MODEL_PROOFS={}
_RUNTIME=Path(EXPECTED['resource_root'])
for _key,_folder in {'RAG_V2_CACHE_DIR':'indexes','TORCH_HOME':'model_cache','XDG_CACHE_HOME':'model_cache','RAG_V2_LOG_DIR':'logs','RAG_V2_TRACE_DIR':'traces','RAG_V2_EVALUATION_DIR':'evaluation'}.items():
 os.environ[_key]=str(_RUNTIME/_folder)
def require_target(service):
 keys=['MYSQL_HOST','MYSQL_PORT','MYSQL_DATABASE','RAG_V2_MYSQL_UUID','MYSQL_USER','MYSQL_PASSWORD'] if service=='mysql' else ['REDIS_HOST','REDIS_PORT','REDIS_DB','RAG_V2_REDIS_PREFIX','REDIS_PASSWORD','RAG_V2_REDIS_IDENTITY']
 if any(not os.environ.get(k) or os.environ[k]!=str(_PRIVATE[k]) for k in keys):raise RuntimeError('Isolated '+service+' target mismatch')
 if service=='mysql' and (os.environ['MYSQL_HOST']!=('mysql' if _CONTAINER else '127.0.0.1') or os.environ['MYSQL_PORT']!='43317' or not os.environ['MYSQL_DATABASE'].startswith('rag_v2_docker_' if _CONTAINER else 'rag_v2_v3_audit_')):raise RuntimeError('Forbidden database target')
 if service=='redis' and (os.environ['REDIS_PORT']!='46397' or os.environ['REDIS_DB']=='0' or not os.environ['RAG_V2_REDIS_PREFIX'].startswith('ragv2v3_audit_')):raise RuntimeError('Forbidden Redis target')
def verify_mysql(connection):
 try:
  with connection.cursor() as cursor:
   cursor.execute('SELECT @@port,DATABASE(),@@server_uuid,@@datadir')
   row=cursor.fetchone();values=tuple(row.values()) if isinstance(row,dict) else row
   expected_dir=EXPECTED['mysql_datadir']
   if values[:3]!=(43317,_PRIVATE['MYSQL_DATABASE'],_PRIVATE['RAG_V2_MYSQL_UUID']) or Path(values[3]).resolve()!=Path(expected_dir).resolve():raise RuntimeError('Connected MySQL identity mismatch')
  connection.rollback();return connection
 except BaseException:connection.close();raise
# Prevent Python filesystem writes into protected models and original project/data roots.
_PROTECTED=[Path(p).resolve() for p in EXPECTED['protected_roots']]
def _audit(event,args):
 paths=[]
 if event=='open':
  path,mode,flags=args
  if isinstance(path,(str,bytes,os.PathLike)) and ((mode and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):paths=[path]
 elif event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.utime']:paths=[args[0]]
 elif event in ['os.rename','os.link','os.symlink']:paths=list(args[:2])
 for path in paths:
  if isinstance(path,(str,bytes,os.PathLike)):
   resolved=Path(os.fsdecode(path)).resolve()
   if any(resolved==p or p in resolved.parents for p in _PROTECTED):raise PermissionError('Protected original resource write denied')
sys.addaudithook(_audit)
def readonly_model_path(variable):
 p=Path(os.environ[variable]).resolve(strict=True)
 if str(p)!=str(Path(EXPECTED[variable]).resolve()) or not p.is_dir():raise RuntimeError('Model path mismatch')
 import hashlib
 rows=json.loads((Path('/run/secrets/models.json') if _CONTAINER else ROOT/'.isolation/models.json').read_text())
 selected=[r for r in rows if p in Path(r['path']).parents]
 if not selected:raise RuntimeError('Missing model fingerprint')
 for row in selected:
  with Path(row['path']).open('rb') as stream:
   checksum=hashlib.file_digest(stream,'sha256').hexdigest()
  if checksum!=row['sha256']:raise RuntimeError('Model content changed')
 from model_fingerprint import model_receipt
 name={'RAG_V2_EMBED_PATH':'BAAI/bge-small-zh-v1.5','RAG_V2_RERANK_PATH':'BAAI/bge-reranker-base'}[variable]
 _MODEL_PROOFS[variable]=model_receipt(name,p,selected)
 return p

def verified_model_identity(variable):
 """Relocate identical read-only model bytes without rewriting snapshots."""
 path=readonly_model_path(variable)
 if not _CONTAINER:return str(path)
 identities=EXPECTED.get('model_identities',{})
 identity=identities.get(variable)
 if not identity:raise RuntimeError('Missing verified relocated model identity')
 return identity

def verified_model_metadata(variable):
 proof=_MODEL_PROOFS.get(variable)
 if not proof or Path(os.environ[variable]).resolve()!=Path(proof['location']):
  raise RuntimeError('Model bytes must be verified before recording their identity')
 return dict(proof)
