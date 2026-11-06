"""Request-scoped atomic execution accounting and cooperative cancellation."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from time import perf_counter

current = ContextVar("rag_execution_budget", default=None)
cancel_signal = ContextVar("rag_cancellation", default=None)


class BudgetExceeded(TimeoutError):
    pass


class ExecutionBudget:
    def __init__(self, options):
        self.timeout_ms = options.timeout_budget_ms
        self.token_limit = options.token_budget
        self.started = perf_counter()
        self.deadline = self.started + options.timeout_budget_ms / 1000
        self.limit = options.max_tool_calls
        self.model_limit = options.max_model_calls
        self.graph_limit = options.max_graph_queries
        self.used = self.models = self.graphs = 0
        self.events = []
        self.reason = None
        self.lock = Lock()
        self.cancelled = cancel_signal.get()
        self.bounds={"max_iterations":options.max_retrieval_attempts,"max_sub_queries":options.max_sub_queries,"max_graph_hops":options.max_graph_hops,"max_final_evidence":options.max_final_evidence,"max_retries":0}

    def acquire(self, name, kind):
        with self.lock:
            reason = ("cancelled" if self.cancelled is not None and self.cancelled.is_set() else
                      "deadline" if perf_counter() >= self.deadline else
                      "tool_budget" if kind != "local" and self.used >= self.limit else
                      "model_budget" if kind == "model" and self.models >= self.model_limit else
                      "graph_budget" if kind == "graph" and self.graphs >= self.graph_limit else None)
            if reason:
                self.reason = reason
                self.events.append({"stage":name,"status":"blocked","reason":reason,"ms":0})
                raise BudgetExceeded(reason)
            if kind != "local":
                self.used += 1
                self.models += kind == "model"
                self.graphs += kind == "graph"
            event = {"stage":name,"kind":kind,"status":"started","call_number":self.used,"start":perf_counter()}
            self.events.append(event)
            return event

    def snapshot(self):
        terminal = self.reason or ("cancelled" if self.cancelled is not None and self.cancelled.is_set() else "deadline" if perf_counter() >= self.deadline else "completed")
        return {"initial":{"max_tool_calls":self.limit,"max_model_calls":self.model_limit,"max_graph_queries":self.graph_limit,"timeout_budget_ms":self.timeout_ms,"estimated_token_limit":self.token_limit},
                "elapsed_ms":round((perf_counter()-self.started)*1000,3), "used_tool_calls":self.used,"used_model_calls":self.models,"used_graph_queries":self.graphs,
                "remaining_tool_calls":self.limit-self.used,"termination_reason":terminal,
                "bounds":self.bounds,"events":[{k:v for k,v in e.items() if k!="start"} for e in self.events]}

def checkpoint(deadline=None):
    """Cooperative guard: does not claim to interrupt native/model computations."""
    budget = current.get()
    reason = ('cancelled' if budget and budget.cancelled is not None and budget.cancelled.is_set() else
              'deadline' if (budget and perf_counter() >= budget.deadline) else None)
    if reason:
        budget.reason = reason
        raise BudgetExceeded(reason)
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError('stage_deadline')


@contextmanager
def stage(name, kind="local"):
    budget = current.get()
    event = budget.acquire(name,kind) if budget is not None else None
    try:
        yield
        checkpoint()
        if event is not None:event["status"]="completed"
    except BaseException as exc:
        if event is not None:event.update(status="failed",error=type(exc).__name__)
        raise
    finally:
        if event is not None:event["ms"]=round((perf_counter()-event["start"])*1000,3)


def request_scoped(function):
    @wraps(function)
    def wrapper(store,view,question,history,options,*args,**kwargs):
        budget=current.get() or ExecutionBudget(options)
        token=current.set(budget)
        try:
            result=function(store,view,question,history,options,*args,**kwargs)
            if result.get('answer_action') in {'ANSWER','CLARIFY','CONFLICT'}:
                checkpoint()
                # Optional-stage denial remains in events, not as a false global failure.
                budget.reason = None
            elif result.get('answer_action')=='ABSTAIN' and budget.reason is None and budget.used>=budget.limit:
                budget.reason='tool_budget'
            graph_events=[]
            sub_events=[]
            for attempt in result.get("retrieval_attempts",[]):
                trace=attempt.get("retrieval",attempt)
                graph_events.extend(r for r in trace.get("routes",[]) if r.get("id")=="graph")
                sub_events.extend(r for r in trace.get("graph_subqueries",[])
                                  if r.get('retrieval_status') in {'used','failed'})
            for entry in budget.events:
                routes=graph_events if entry.get("stage")=="graph" else sub_events
                if entry.get("status")=="scheduled" and entry.get("stage") in {"graph","graph_subquery"} and routes:
                    record=routes.pop(0)
                    status=record.get("status","completed")
                    status={'COMPLETED':'completed','FAILED':'failed','TIMEOUT':'failed'}.get(status,status)
                    entry.update(status=status,ms=record.get("ms") if record.get("ms") is not None else record.get('elapsed_ms',0))
            result.setdefault("budget",{})["execution"]=budget.snapshot()
            return result
        finally:current.reset(token)
    return wrapper


def tracked(name,kind="local"):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with stage(name,kind):
                return function(*args,**kwargs)
        return wrapped
    return decorate
