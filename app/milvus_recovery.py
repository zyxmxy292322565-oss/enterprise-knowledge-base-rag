"""Explicit, hash-bound migration recovery; never repairs the source in place."""
from pathlib import Path
import re
from knowledge_v2 import read_checked,KnowledgeError
from request_budget import BudgetExceeded

def verified_vector_bytes(folder,manifest,recovery_root,allow_recovery=False):
    folder=Path(folder);sid=manifest['snapshot_id'];expected=manifest['index_sha']
    if not re.fullmatch('[a-f0-9]{32}',sid) or folder.name!=sid or not re.fullmatch('[a-f0-9]{64}',expected):
        raise ValueError('Invalid recovery identity')
    try:return read_checked(folder/'index.faiss',expected),False
    except BudgetExceeded:raise
    except (KnowledgeError,OSError):
        if not allow_recovery:raise
    backup=Path(recovery_root)/sid/'index.faiss'
    return read_checked(backup,expected),True
