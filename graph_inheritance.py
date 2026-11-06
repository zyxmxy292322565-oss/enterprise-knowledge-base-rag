"""Transactional, evidence-bound inheritance; indexed matching and bounded SQL batches."""
import json
from collections import defaultdict
from hashlib import sha256
from time import perf_counter
from knowledge_v2 import KnowledgeError, json_bytes
from graph_governance import draft
from request_budget import checkpoint

def _proof_key(row, members):
    member=members.get(row['document_version'])
    if not member:return None
    return json_bytes([member['source'],row['document_id'],member['content_hash'],member['policy_version'],
        row['document_version'],row['chunk_id'],row['parent_id'],row['file_hash'],
        json.loads(row['source_position']),row['evidence_text']])

def packed(value):return json.dumps(value,ensure_ascii=False,default=str)

def inherit(store, source, target, actor):
    if source==target:raise KnowledgeError('Inheritance requires different snapshots')
    start=perf_counter();old=store.load_snapshot(source,strict_sparse=True);new=store.load_snapshot(target,strict_sparse=True)
    old_members={m['document_version_id']:m for m in old.manifest['members']}
    new_members={m['document_version_id']:m for m in new.manifest['members']}
    run_id=sha256(json_bytes([source,target])).hexdigest();records=[];scanned=0;queries=0
    with store.transaction() as cursor:
        draft(cursor,store,target)
        def fetch(sql,args=()):
            nonlocal scanned,queries
            checkpoint();queries+=1;cursor.execute(sql,args);rows=cursor.fetchall();scanned+=len(rows);return rows
        prior=fetch('SELECT state FROM kg_inheritance_runs WHERE run_id=%s',(run_id,))
        if prior:
            if prior[0]['state']!='COMPLETED':raise KnowledgeError('Incomplete inheritance requires investigation',409)
            return {'source':source,'target':target,'run_id':run_id,'idempotent':True,
                    'records':fetch('SELECT kind,source_object,status FROM kg_inheritance_objects WHERE run_id=%s',(run_id,))}
        job=fetch('SELECT task_status FROM kg_builds WHERE snapshot_id=%s',(source,))
        if not job or job[0]['task_status']!='SEALED':raise KnowledgeError('Inheritance source must be sealed',409)
        cursor.execute("INSERT INTO kg_inheritance_runs(run_id,source_snapshot,target_snapshot,actor,state,metrics) VALUES(%s,%s,%s,%s,'RUNNING',JSON_OBJECT())",(run_id,source,target,actor))
        reviews=fetch('SELECT * FROM kg_reviews WHERE snapshot_id=%s ORDER BY review_id',(source,))
        review_by_id={r['review_id']:r for r in reviews}
        lineage=defaultdict(list);alias_reviews={}
        for row in reviews:
            lineage[(row['entity_kind'],row['object_id'])].append(row['review_id'])
            if row['entity_kind']=='alias':
                try:
                    note=json.loads(row['note'])
                    if note.get('alias'):alias_reviews[(row['object_id'],note['alias'])]=(row,note)
                except (ValueError,TypeError):pass
        old_proofs=fetch('SELECT * FROM kg_relation_evidence WHERE snapshot_id=%s',(source,))
        new_proofs=fetch('SELECT * FROM kg_relation_evidence WHERE snapshot_id=%s',(target,))
        proof_index=defaultdict(list);old_by_edge=defaultdict(list);new_by_edge=defaultdict(list)
        for row in new_proofs:
            checkpoint();proof_index[(row['relation_id'],_proof_key(row,new_members))].append(row);new_by_edge[row['relation_id']].append(row)
        proof_map={};binding_rows=[]
        proof_relations=defaultdict(set);proof_quotes={}
        for row in old_proofs:
            checkpoint();old_by_edge[row['relation_id']].append(row)
            proof_relations[row['evidence_id']].add(row['relation_id']);proof_quotes[(row['relation_id'],row['evidence_id'])]=row['evidence_text']
            matches=proof_index.get((row['relation_id'],_proof_key(row,old_members)),[])
            status='AUTO_REBOUND' if len(matches)==1 else 'NEEDS_REVIEW' if new_by_edge[row['relation_id']] else 'REJECTED'
            other=matches[0] if len(matches)==1 else None
            if other:proof_map[(row['relation_id'],row['evidence_id'])]=other['evidence_id']
            binding_rows.append((run_id,row['relation_id'],row['evidence_id'],other['evidence_id'] if other else None,status,
                packed({'source_child':row['chunk_id'],'source_document_version':row['document_version'],
                  'source_document_id':row['document_id'],'source_file_hash':row['file_hash'],
                  'source_name':old_members.get(row['document_version'],{}).get('source'),
                  'source_position':json.loads(row['source_position']),'source_content_hash':old_members.get(row['document_version'],{}).get('content_hash'),
                  'target_child':other['chunk_id'] if other else None,'quote_sha256':sha256(row['evidence_text'].encode()).hexdigest()})))
        old_edges=fetch("SELECT * FROM kg_relations WHERE snapshot_id=%s AND status='APPROVED'",(source,))
        revoked={r['relation_id'] for r in fetch('SELECT relation_id FROM kg_policy_revocations WHERE snapshot_id=%s',(source,))}
        new_edges={r['relation_id']:r for r in fetch('SELECT * FROM kg_relations WHERE snapshot_id=%s',(target,))}
        old_sem={r['relation_id']:(r['polarity'],json.loads(r['conditions_json']),json.loads(r['exceptions_json'])) for r in fetch('SELECT * FROM kg_relation_semantics WHERE snapshot_id=%s',(source,))}
        new_sem={r['relation_id']:(r['polarity'],json.loads(r['conditions_json']),json.loads(r['exceptions_json'])) for r in fetch('SELECT * FROM kg_relation_semantics WHERE snapshot_id=%s',(target,))}
        safe_edges=set();incident=defaultdict(set)
        keys=('source_entity_id','target_entity_id','relation_type','valid_from','valid_to')
        for edge in old_edges:
            checkpoint();key=edge['relation_id'];other=new_edges.get(key);proofs=old_by_edge[key]
            incident[edge['source_entity_id']].add(key);incident[edge['target_entity_id']].add(key)
            if (key not in revoked and other and other['status']!='REJECTED' and proofs and all((key,p['evidence_id']) in proof_map for p in proofs)
                and all(edge[k]==other[k] for k in keys) and key in old_sem and old_sem[key]==new_sem.get(key)):safe_edges.add(key)
        old_entities=fetch("SELECT * FROM kg_entities WHERE snapshot_id=%s AND status='APPROVED'",(source,))
        new_entities={r['entity_id']:r for r in fetch('SELECT * FROM kg_entities WHERE snapshot_id=%s',(target,))}
        safe_entities=set()
        for e in old_entities:
            checkpoint();key=e['entity_id'];other=new_entities.get(key)
            if other and other['status']!='REJECTED' and incident[key] and incident[key]<=safe_edges and all(e[k]==other[k] for k in ('entity_type','normalized_name','description')):safe_entities.add(key)
        def audit(kind,key,status,detail=None,review_ids=None):
            records.append({'kind':kind,'source_object':key,'status':status,'detail':detail or {},
                'source_review_ids':review_ids if review_ids is not None else lineage[(kind,key)]})
        for e in old_entities:audit('entity',e['entity_id'],'AUTO_REBOUND' if e['entity_id'] in safe_entities else 'NEEDS_REVIEW',{'previous_status':new_entities.get(e['entity_id'],{}).get('status')})
        for edge in old_edges:
            key=edge['relation_id']
            if edge['source_entity_id'] not in safe_entities or edge['target_entity_id'] not in safe_entities:safe_edges.discard(key)
            audit('relation',key,'AUTO_REBOUND' if key in safe_edges else 'NEEDS_REVIEW' if key in new_edges else 'REJECTED',{'previous_status':new_edges.get(key,{}).get('status')})
        for inherited in fetch('''SELECT o.* FROM kg_inheritance_objects o JOIN kg_inheritance_runs r ON r.run_id=o.run_id
            WHERE r.target_snapshot=%s AND o.status='AUTO_REBOUND' ''',(source,)):
            detail=json.loads(inherited['details']);event=review_by_id.get(inherited['review_id'])
            if event and event['status']=='AUTO_REBOUND':
                lineage[(inherited['kind'],inherited['target_object'])].append(inherited['review_id'])
                if inherited['kind']=='alias':
                    alias_reviews.setdefault((detail['entity_id'],detail['alias']),
                        ({'review_id':inherited['review_id'],'status':'APPROVED'}, {'evidence_id':detail['target_evidence_id']}))
        target_aliases={(r['entity_id'],r['alias']) for r in fetch('SELECT entity_id,alias FROM kg_entity_aliases WHERE snapshot_id=%s',(target,))}
        alias_inserts=[]
        for alias in fetch('SELECT * FROM kg_entity_aliases WHERE snapshot_id=%s',(source,)):
            key=sha256(json_bytes([alias['entity_id'],alias['alias']])).hexdigest();event=alias_reviews.get((alias['entity_id'],alias['alias']))
            canonical=new_entities.get(alias['entity_id'],{}).get('normalized_name')==alias['alias']
            proof_id=event[1].get('evidence_id') if event else None
            alias_bindings={proof_map[(edge,proof_id)] for edge in proof_relations[proof_id]&incident[alias['entity_id']]
                if (edge,proof_id) in proof_map and alias['alias'] in proof_quotes.get((edge,proof_id),'').casefold()}
            safe=alias['entity_id'] in safe_entities and (canonical or bool(event and event[0]['status']=='APPROVED'
                and len(alias_bindings)==1))
            if safe:alias_inserts.append((target,alias['entity_id'],alias['alias']))
            if event:audit('alias',key,'AUTO_REBOUND' if safe else 'NEEDS_REVIEW',{'entity_id':alias['entity_id'],'alias':alias['alias'],'source_evidence_id':proof_id,'target_evidence_id':next(iter(alias_bindings)) if len(alias_bindings)==1 else None,'target_existed':(alias['entity_id'],alias['alias']) in target_aliases},[event[0]['review_id']])
        old_merges=fetch('SELECT * FROM kg_entity_merges WHERE snapshot_id=%s',(source,))
        target_merges={r['source_id']:r for r in fetch('SELECT * FROM kg_entity_merges WHERE snapshot_id=%s AND undone_at IS NULL',(target,))}
        merge_inserts=[]
        for item in old_merges:
            safe=item['source_id'] in safe_entities and item['target_id'] in safe_entities and item['undone_at'] is None
            other=target_merges.get(item['source_id'])
            if other:safe=safe and all(item[k]==other[k] for k in ('operation_id','source_id','target_id','undone_at'))
            if safe and not other:
                merge_inserts.append((target,*[item[k] for k in ('operation_id','source_id','target_id','actor','reason','created_at','undone_at','undone_by','undo_reason')]))
                target_merges[item['source_id']]=item
            audit('merge',item['operation_id'],'AUTO_REBOUND' if safe else 'REJECTED' if item['undone_at'] else 'NEEDS_REVIEW',{'undone_at':item['undone_at']})
        policies=fetch('SELECT * FROM kg_policy_versions WHERE snapshot_id=%s',(source,))
        target_policies={r['relation_id']:r for r in fetch('SELECT * FROM kg_policy_versions WHERE snapshot_id=%s',(target,))}
        policy_inserts=[]
        supersession_bindings=defaultdict(set)
        edge_types={e['relation_id']:e['relation_type'] for e in old_edges}
        for (edge,proof),target_proof in proof_map.items():
            if edge_types.get(edge)=='SUPERSEDES':supersession_bindings[proof].add(target_proof)
        def comparable(row):
            from graph_governance import Policy
            meta=json.loads(row['metadata_json']);return {k:meta.get(k) for k in Policy.model_fields if k!='reason'}
        for policy in policies:
            checkpoint();meta=json.loads(policy['metadata_json']);key=policy['relation_id']
            safe=key in safe_edges and (key,policy['evidence_id']) in proof_map and policy['policy_entity_id'] in safe_entities and key not in revoked and meta.get('policy_status')=='ACTIVE'
            if meta.get('supersedes_relation_id'):safe=safe and meta['supersedes_relation_id'] in safe_edges and len(supersession_bindings[meta.get('supersession_proof_id')])==1
            if safe:
                meta['evidence_id']=proof_map[(key,policy['evidence_id'])]
                if meta.get('supersession_proof_id'):meta['supersession_proof_id']=next(iter(supersession_bindings[meta['supersession_proof_id']]))
                meta['inherited_from']=source
                row=policy|{'evidence_id':meta['evidence_id'],'metadata_json':packed(meta)};other=target_policies.get(key)
                if other and (any(row[k]!=other[k] for k in ('policy_id','policy_version','policy_entity_id','evidence_id')) or comparable(row)!=comparable(other)):safe=False
                elif not other:policy_inserts.append((target,key,policy['policy_id'],policy['policy_version'],policy['policy_entity_id'],meta['evidence_id'],packed(meta)))
            audit('policy',key,'AUTO_REBOUND' if safe else 'REJECTED' if key in revoked else 'NEEDS_REVIEW',{'target_conflict':bool(key in target_policies and not safe),'target_existed':key in target_policies})
        def batch(sql,rows):
            nonlocal queries
            for offset in range(0,len(rows),500):checkpoint();queries+=1;cursor.executemany(sql,rows[offset:offset+500])
        for table,column,ids in [('kg_entities','entity_id',safe_entities),('kg_relations','relation_id',safe_edges)]:
            ids=sorted(ids)
            for offset in range(0,len(ids),500):
                group=ids[offset:offset+500];queries+=1;cursor.execute('UPDATE '+table+" SET status='APPROVED' WHERE snapshot_id=%s AND "+column+' IN ('+','.join(['%s']*len(group))+')',(target,*group))
        batch('INSERT INTO kg_inheritance_bindings VALUES(%s,%s,%s,%s,%s,%s)',binding_rows)
        batch('INSERT IGNORE INTO kg_entity_aliases VALUES(%s,%s,%s)',alias_inserts)
        batch('INSERT INTO kg_entity_merges VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',merge_inserts)
        batch('INSERT INTO kg_policy_versions VALUES(%s,%s,%s,%s,%s,%s,%s)',policy_inserts)
        review_rows=[]
        for record in records:
            event=sha256(json_bytes([source,record['kind'],record['source_object']])).hexdigest();record['object_id']=event
            note={'source_snapshot':source,'target_snapshot':target,'run_id':run_id,'kind':record['kind'],'source_object':record['source_object']}
            review_rows.append((target,'inheritance',event,'PENDING',record['status'],actor,packed(note)))
        # All VALUES are bound parameters so PyMySQL uses its multi-row INSERT path.
        batch("INSERT INTO kg_reviews(snapshot_id,entity_kind,object_id,previous_status,status,actor,note) VALUES(%s,%s,%s,%s,%s,%s,%s)",review_rows)
        events=fetch("SELECT object_id,review_id FROM kg_reviews WHERE snapshot_id=%s AND entity_kind='inheritance'",(target,));event_ids={r['object_id']:r['review_id'] for r in events}
        batch('INSERT INTO kg_inheritance_objects VALUES(%s,%s,%s,%s,%s,%s,%s,%s)',[(run_id,r['kind'],r['source_object'],r['source_object'],r['status'],packed(r['source_review_ids']),event_ids[r['object_id']],packed(r['detail'])) for r in records])
        metrics={'application_query_batches':queries,'fetched_rows':scanned,'proof_bindings':len(binding_rows),'objects':len(records),'elapsed_ms':round((perf_counter()-start)*1000,3)}
        cursor.execute("UPDATE kg_inheritance_runs SET state='COMPLETED',metrics=%s WHERE run_id=%s",(packed(metrics),run_id))
    return {'source':source,'target':target,'run_id':run_id,'records':[{k:r[k] for k in ('object_id','kind','status')} for r in records],'metrics':metrics}
