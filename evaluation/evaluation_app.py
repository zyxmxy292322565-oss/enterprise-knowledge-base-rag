"""Evaluation-only application. Never import this module in the production app."""
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Literal


def validate_isolation():
    required = {
        "EVALUATION_ISOLATED": "1", "EVALUATION_CACHE_DIR": "/app/cache",
        "EVALUATION_MYSQL_HOST": "mysql", "EVALUATION_REDIS_HOST": "redis",
        "MYSQL_HOST": "mysql", "MYSQL_PORT": "3306",
        "MYSQL_DATABASE": "rag_evaluation", "MYSQL_USER": "rag_eval",
        "REDIS_HOST": "redis", "REDIS_PORT": "6379", "REDIS_DB": "0",
    }
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise RuntimeError("Evaluation isolation configuration missing or invalid")
    if not os.environ.get("MYSQL_PASSWORD") or not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Evaluation credentials are required")
    if Path.cwd().resolve() != Path('/app') or Path('cache').resolve() != Path('/app/cache'):
        raise RuntimeError("Evaluation cache path is not /app/cache")


# Fail before importing modules that read configuration or initialise services.
validate_isolation()

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import api_app
import models
from evaluation_tracking import runtime_manifest
from db_utils import get_connection


class EvaluationLLM:
    """Evaluation-only LLM wrapper with call, token and budget accounting."""

    def __init__(self):
        if os.environ.get('DEEPSEEK_MODEL') != 'deepseek-flash':
            raise RuntimeError('Evaluation answer runs require DEEPSEEK_MODEL=deepseek-flash')
        if os.environ.get('EVALUATION_LLM_NON_THINKING') != '1':
            raise RuntimeError('Evaluation answer runs require non-thinking mode')
        self.max_calls = int(os.environ.get('EVALUATION_LLM_MAX_CALLS', '132'))
        self.max_tokens = int(os.environ.get('EVALUATION_LLM_MAX_TOKENS', '800'))
        self.budget_cny = float(os.environ.get('EVALUATION_LLM_BUDGET_CNY', '1.50'))
        self.input_cny_per_mtok = float(os.environ.get('EVALUATION_LLM_INPUT_CNY_PER_MTOK', '2'))
        self.output_cny_per_mtok = float(os.environ.get('EVALUATION_LLM_OUTPUT_CNY_PER_MTOK', '8'))
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost_cny = 0.0
        self.last_call = None
        self.client = ChatOpenAI(
            model='deepseek-flash',
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url=os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com'),
            max_tokens=self.max_tokens,
            max_retries=0,
            extra_body={'thinking': {'type': 'disabled'}},
        )

    @staticmethod
    def _usage(response):
        usage = dict(getattr(response, 'usage_metadata', None) or {})
        metadata = dict(getattr(response, 'response_metadata', None) or {})
        token_usage = dict(metadata.get('token_usage') or {})
        input_tokens = usage.get('input_tokens', token_usage.get('prompt_tokens'))
        output_tokens = usage.get('output_tokens', token_usage.get('completion_tokens'))
        total_tokens = usage.get('total_tokens', token_usage.get('total_tokens'))
        if input_tokens is None or output_tokens is None:
            raise RuntimeError('LLM response did not include token usage')
        return {
            'input_tokens': int(input_tokens),
            'output_tokens': int(output_tokens),
            'total_tokens': int(total_tokens or (input_tokens + output_tokens)),
            'input_token_details': usage.get('input_token_details'),
            'output_token_details': usage.get('output_token_details'),
        }

    def invoke(self, prompt):
        if self.call_count >= self.max_calls:
            raise RuntimeError('Evaluation LLM call limit reached')
        if self.estimated_cost_cny >= self.budget_cny:
            raise RuntimeError('Evaluation LLM budget reached')
        self.call_count += 1
        attempt = self.call_count
        try:
            response = self.client.invoke(prompt)
            usage = self._usage(response)
            self.total_input_tokens += usage['input_tokens']
            self.total_output_tokens += usage['output_tokens']
            self.estimated_cost_cny = (
                self.total_input_tokens * self.input_cny_per_mtok
                + self.total_output_tokens * self.output_cny_per_mtok
            ) / 1_000_000
            self.last_call = {
                'attempt': attempt,
                'status': 'success',
                'model': 'deepseek-flash',
                'thinking_mode': 'disabled',
                'max_tokens': self.max_tokens,
                'usage': usage,
                'cumulative_input_tokens': self.total_input_tokens,
                'cumulative_output_tokens': self.total_output_tokens,
                'cumulative_estimated_cost_cny': round(self.estimated_cost_cny, 8),
                'budget_cny': self.budget_cny,
            }
            if self.estimated_cost_cny > self.budget_cny:
                raise RuntimeError('Evaluation LLM budget exceeded')
            return response
        except Exception as exc:
            if not self.last_call or self.last_call.get('attempt') != attempt:
                self.last_call = {
                    'attempt': attempt,
                    'status': 'error',
                    'error_type': type(exc).__name__,
                    'model': 'deepseek-flash',
                    'thinking_mode': 'disabled',
                    'max_tokens': self.max_tokens,
                    'cumulative_input_tokens': self.total_input_tokens,
                    'cumulative_output_tokens': self.total_output_tokens,
                    'cumulative_estimated_cost_cny': round(self.estimated_cost_cny, 8),
                    'budget_cny': self.budget_cny,
                }
            raise


def create_tables():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS evaluation_experiments (
                id CHAR(36) PRIMARY KEY, dataset_version VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS evaluation_runs (
                id CHAR(36) PRIMARY KEY, experiment_id CHAR(36) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES evaluation_experiments(id))""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS evaluation_case_runs (
                id CHAR(36) PRIMARY KEY, run_id CHAR(36) NOT NULL,
                case_id VARCHAR(255) NOT NULL, record_json LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_run_case (run_id, case_id),
                FOREIGN KEY (run_id) REFERENCES evaluation_runs(id))""")
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app):
    validate_isolation()
    # Connectivity and evaluation schema must succeed before any RAG startup.
    create_tables()
    async with api_app.lifespan(api_app.app):
        # Replace the production-configured client only inside the isolated
        # evaluation process. Public business services never import this app.
        api_app.app.state.chat_model = EvaluationLLM()
        yield


app = FastAPI(lifespan=lifespan)


class ExperimentRequest(BaseModel):
    dataset_version: str = Field(min_length=1, max_length=255)


class CaseRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=255)
    question: str = Field(min_length=1)
    chain: Literal['ask', 'policy'] = 'ask'
    final_k: int = Field(default=3, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    faiss_multiplier: int = Field(default=2, ge=1)
    min_rerank_score: float = 0.2
    rerank_enabled: bool = True
    max_context_chars: int = Field(default=6000, ge=1)
    max_chunk_chars: int = Field(default=1200, ge=1)
    cache_mode: str = Field(default="bypass", pattern="^bypass$")


def _execute(sql, params=(), fetch=False):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone() if fetch else None
        if not fetch:
            conn.commit()
        return row
    finally:
        conn.close()


def snapshot_manifest():
    manifest = runtime_manifest(
        app_state=api_app.app.state,
        prompt_callable=api_app._ask_question,
        source_root=Path('/app'),
        embedding_model=models.EMBED_MODEL_NAME,
        rerank_model=models.RERANK_MODEL_NAME,
        llm_model=os.environ.get('DEEPSEEK_MODEL'),
    )
    # Preserve the B3 record shape while sharing the v2 hashing implementation.
    manifest['source_sha256'] = manifest['source_files_sha256']
    manifest['index_files_sha256'] = {
        name: item['sha256'] for name, item in manifest['index_files'].items()
    }
    return manifest


@app.post('/evaluation/experiments')
def create_experiment(request: ExperimentRequest):
    experiment_id = str(uuid4())
    _execute('INSERT INTO evaluation_experiments (id, dataset_version) VALUES (%s, %s)',
             (experiment_id, request.dataset_version))
    return {'experiment_id': experiment_id}


@app.post('/evaluation/experiments/{experiment_id}/runs')
def create_run(experiment_id: str):
    if not _execute('SELECT id FROM evaluation_experiments WHERE id=%s', (experiment_id,), True):
        raise HTTPException(404, 'Experiment not found')
    run_id = str(uuid4())
    _execute('INSERT INTO evaluation_runs (id, experiment_id) VALUES (%s, %s)', (run_id, experiment_id))
    return {'experiment_id': experiment_id, 'run_id': run_id}


@app.post('/evaluation/runs/{run_id}/cases')
def run_case(run_id: str, request: CaseRequest):
    row = _execute('SELECT e.id, e.dataset_version FROM evaluation_runs r JOIN evaluation_experiments e ON r.experiment_id=e.id WHERE r.id=%s', (run_id,), True)
    if not row:
        raise HTTPException(404, 'Run not found')
    if _execute('SELECT id FROM evaluation_case_runs WHERE run_id=%s AND case_id=%s', (run_id, request.case_id), True):
        raise HTTPException(409, 'Case already exists in run')
    if request.candidate_k is not None and request.candidate_k < request.final_k:
        raise HTTPException(422, 'candidate_k must be at least final_k')
    manifest = snapshot_manifest()
    trace = {}
    options = {'candidate_k': request.candidate_k, 'rerank_enabled': request.rerank_enabled,
               'cache_mode': request.cache_mode}
    if request.chain == 'ask':
        ask = api_app.AskRequest(question=request.question, top_k=request.final_k,
            faiss_multiplier=request.faiss_multiplier, min_rerank_score=request.min_rerank_score,
            max_context_chars=request.max_context_chars, max_chunk_chars=request.max_chunk_chars)
        result = api_app._ask_question(ask, evaluation_options=options, trace=trace)
        trace['llm_call'] = api_app.app.state.chat_model.last_call
        status = trace.get('status', 'error')
    else:
        policy = api_app.PolicyRAGRequest(question=request.question, top_k=request.final_k,
            faiss_multiplier=request.faiss_multiplier, min_rerank_score=request.min_rerank_score)
        result = api_app._query_policy_rag(policy, evaluation_options=options, trace=trace)
        status = ('error' if not result.get('success') else
                  'refused_low_relevance' if result.get('refused') else 'evidence_returned')
    # Avoid persisting exception text, which may include provider credentials.
    if status == 'error':
        result = {'answer': None, 'code': result.get('code', 'EVALUATION_FAILED'),
                  'message': 'evaluation execution failed', 'timing': result.get('timing', {})}
    elif not request.rerank_enabled:
        # The production serializers expect a numeric score. Do not report
        # the internal placeholder as a real rerank score in experiment data.
        for item in result.get('sources', []) + result.get('evidence', []):
            item['rerank_score'] = None
    after = snapshot_manifest()
    if manifest['index_files_sha256'] != after['index_files_sha256']:
        status = 'invalid_index_changed'
    case_run_id = str(uuid4())
    record = {'experiment_id': row[0], 'run_id': run_id, 'case_run_id': case_run_id,
              'case_id': request.case_id, 'dataset_version': row[1], 'manifest': manifest,
              'requested_parameters': request.model_dump(exclude={'question'}), 'question': request.question,
              'status': status, 'trace': trace, 'result': result}
    _execute('INSERT INTO evaluation_case_runs (id, run_id, case_id, record_json) VALUES (%s, %s, %s, %s)',
             (case_run_id, run_id, request.case_id, json.dumps(record, ensure_ascii=False, allow_nan=False)))
    return record


@app.get('/evaluation/cases/{case_run_id}')
def get_case(case_run_id: str):
    row = _execute('SELECT record_json FROM evaluation_case_runs WHERE id=%s', (case_run_id,), True)
    if not row:
        raise HTTPException(404, 'Case run not found')
    return json.loads(row[0])


@app.post('/evaluation/documents')
async def upload_document(file: UploadFile = File(...)):
    # Upload is confined to the evaluation app's guarded DB and index volume.
    return await api_app.upload_file(file, category='evaluation', product_scope='evaluation',
                                     version='evaluation', created_by='evaluation')
