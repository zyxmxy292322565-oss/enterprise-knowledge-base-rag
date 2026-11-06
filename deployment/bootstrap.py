"""Create a fresh local Compose deployment without importing private data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_fingerprint import model_receipt

ROOT = Path(__file__).resolve().parents[1]
PORTS = {"api": 18631, "ui": 18632, "mysql": 43327, "redis": 46407,
         "milvus": 19531, "milvus_health": 19091}
MODEL_NAMES = {"embed": "BAAI/bge-small-zh-v1.5", "rerank": "BAAI/bge-reranker-base"}


def run(command, *, capture=False, stdin=None, env=None):
    return subprocess.run(command, cwd=ROOT, input=stdin, text=True, encoding="utf-8",
                          stdout=subprocess.PIPE if capture else None, check=True, env=env).stdout


def private_write(path, content):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600),
                   "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def fingerprint(model, mount, name):
    rows = []
    if not model.is_dir():
        raise SystemExit(f"Missing local {name} directory")
    for item in sorted(model.rglob("*")):
        if ".git" in item.relative_to(model).parts:
            continue
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise SystemExit("Use a materialized model directory, without linked files")
        if item.is_file():
            with item.open("rb") as stream:
                rows.append(dict(path=f"{mount}/{item.relative_to(model).as_posix()}",
                                 sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    proof = model_receipt(name, mount, rows)
    return rows, f"{name}@{proof['revision']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", required=True, type=Path, help="New private directory outside the checkout")
    parser.add_argument("--reviewer", help="Optional local graph reviewer identity")
    for model in MODEL_NAMES:
        parser.add_argument(f"--{model}", required=True, type=Path)
    for key, default in PORTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}-port", type=int, default=default)
    args = parser.parse_args()
    resource = args.resource.resolve()
    models = {key: getattr(args, key).resolve() for key in MODEL_NAMES}
    if resource.exists() or resource.is_relative_to(ROOT):
        parser.error("--resource must be a new directory outside the checkout")
    if any(resource.is_relative_to(model) or model.is_relative_to(resource) for model in models.values()):
        parser.error("Keep the private resource and read-only model directories separate")
    if any(re.search(r"['\r\n]", str(path)) for path in [resource, *models.values()]):
        parser.error("Paths must not contain quotes or line breaks")
    if args.reviewer is not None and not 1 <= len(args.reviewer.strip()) <= 100:
        parser.error("Provide a reviewer identity of 1–100 characters")
    ports = {key: getattr(args, f"{key}_port") for key in PORTS}
    if len(set(ports.values())) != len(ports) or any(not 1024 <= p <= 65535 for p in ports.values()):
        parser.error("Provide distinct unprivileged ports")
    for port in ports.values():
        try:
            with socket.socket() as sock:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(("127.0.0.1", port))
        except OSError:
            parser.error(f"Port {port} is unavailable; select another port")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("COMPOSE_", "RAG_DOCKER_", "RAG_MILVUS_"))}
    # Keep the caller's Docker context; Docker Compose creates unique project resources.
    run(["docker", "info", "--format", "{{.ServerVersion}}"], capture=True, env=env)
    run_id = secrets.token_hex(6)
    project = f"ragv3-{run_id}"
    milvus_project = f"{project}-milvus"
    rows, model_ids = [], {}
    for key, name in MODEL_NAMES.items():
        model_rows, identity = fingerprint(models[key], f"/models/{key}", name)
        rows.extend(model_rows)
        model_ids[f"RAG_V2_{key.upper()}_PATH"] = identity

    db, prefix = f"rag_v2_docker_{run_id}", f"ragv2v3_audit_docker_{run_id}:"
    mysql_root, mysql_app, redis_pw, redis_id, admin, minio_pw = [secrets.token_urlsafe(32) for _ in range(6)]
    values = dict(RAG_DOCKER_RESOURCE=resource.as_posix(), RAG_DOCKER_DATABASE=db,
                  RAG_DOCKER_EMBED=models["embed"].as_posix(), RAG_DOCKER_RERANK=models["rerank"].as_posix(),
                  RAG_DOCKER_COST_DIR=(resource / "cost").as_posix(), RAG_DOCKER_APP_IMAGE=f"{project}:local",
                  RAG_DOCKER_MILVUS_NETWORK=f"{milvus_project}_default",
                  RAG_MILVUS_MINIO_USER="ragv3_" + run_id, RAG_MILVUS_MINIO_PASSWORD=minio_pw)
    values.update({f"RAG_DOCKER_{key.upper()}_PORT": str(port) for key, port in ports.items()})
    resource.mkdir(mode=0o700)
    if os.name == "nt":
        # Remove inherited access from the private directory, retaining this account.
        sid = re.search(r"S-1-(?:\d+-)+\d+", run(["whoami", "/user", "/fo", "csv", "/nh"], capture=True))
        if not sid:
            raise SystemExit("Unable to identify the current Windows user")
        run(["icacls", str(resource), "/inheritance:r", "/grant:r", f"*{sid[0]}:(OI)(CI)F"], capture=True)
    for directory in ("secrets", "runtime", "evaluation-history", "cost"):
        (resource / directory).mkdir(mode=0o700)
    env_file = resource / "compose.env"
    private_write(env_file, "".join(f"{key}='{value}'\n" for key, value in values.items()))
    sec = resource / "secrets"
    private_write(sec / "mysql_root", mysql_root)
    private_write(sec / "mysql_app", mysql_app)
    private_write(sec / "redis.conf",
                  f"bind 0.0.0.0\nport 46397\nprotected-mode yes\ndir /data\n"
                  f"databases 16\nappendonly yes\nrequirepass {redis_pw}\n")
    for target, data in (("models.json", rows),):
        private_write(sec / target, json.dumps(data))

    def compose(stack, name, *parts, capture=False, stdin=None):
        return run(["docker", "compose", "-p", name, "--env-file", str(env_file),
                    "-f", f"compose.{stack}-isolated.yml", *parts], capture=capture, stdin=stdin, env=env)

    print(f"Projects: {project}, {milvus_project}; private resources: {resource}", flush=True)
    compose("full", project, "up", "-d", "--wait", "--wait-timeout", "180", "mysql", "redis")
    uuid = compose("full", project, "exec", "-T", "mysql", "sh", "-ec",
                   'MYSQL_PWD="$(cat /run/secrets/mysql_app)" mysql -h 127.0.0.1 -P 43317 '
                   '-u "$MYSQL_USER" -N "$MYSQL_DATABASE" -e "SELECT @@server_uuid"',
                   capture=True).strip()
    UUID(uuid)
    marker = compose("full", project, "exec", "-T", "redis", "sh", "-ec",
                     'export REDISCLI_AUTH="$(awk \'$1=="requirepass"{print $2}\' /run/secrets/redis.conf)"; '
                     'redis-cli -p 46397 -n 15 -x SET "$1"', "--", prefix + "instance:identity",
                     capture=True, stdin=redis_id)
    if marker.strip() != "OK":
        raise RuntimeError("Redis identity seed was not acknowledged")
    contract = dict(MYSQL_HOST="mysql", MYSQL_PORT="43317", MYSQL_DATABASE=db, RAG_V2_MYSQL_UUID=uuid,
                    REDIS_HOST="redis", REDIS_PORT="46397", REDIS_DB="15", RAG_V2_REDIS_PREFIX=prefix,
                    RAG_V2_REDIS_IDENTITY=redis_id, RAG_V2_ROOT="/data/store", HF_HOME="/data/HF_HOME",
                    RAG_V2_EMBED_PATH="/models/embed", RAG_V2_RERANK_PATH="/models/rerank")
    uri = "http://standalone:19530"
    identity = dict(contract, deployment_mode="docker", mysql_datadir="/var/lib/mysql/",
                    milvus_uri=uri, resource_root="/data", protected_roots=["/models", "/run/secrets"],
                    model_identities=model_ids)
    private = dict(contract, MYSQL_USER="rag_docker_app", MYSQL_PASSWORD=mysql_app, REDIS_PASSWORD=redis_pw,
                   RAG_V2_ADMIN_TOKEN=admin, RAG_V2_ENABLED="1", RAG_V3_VECTOR_BACKEND="milvus",
                   RAG_V3_MILVUS_URI=uri, RAG_V3_MILVUS_NAMESPACE="ragv3_" + run_id, RAG_V3_GRAPH_ENABLED="1",
                   API_BASE="http://api:8000", RAG_V2_ISOLATED_LLM="1",
                   DEEPSEEK_MODEL="local-fixture", DEEPSEEK_API_KEY="local-fixture",
                   DEEPSEEK_API_BASE="http://127.0.0.1:1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   RAG_V2_MODEL_DEVICE="cpu", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   RAG_V2_CODE_COMMIT=run(["git", "rev-parse", "HEAD"], capture=True).strip())
    if args.reviewer is not None:
        private.update(RAG_V3_GRAPH_REVIEW_ACTOR=args.reviewer.strip(),
                       RAG_V3_GRAPH_REVIEW_TOKEN=secrets.token_urlsafe(32))
    for key, data in (("identity.json", identity), ("environment.json", private)):
        private_write(sec / key, json.dumps(data))
    private_write(resource / "projects.json", json.dumps(dict(app=project, milvus=milvus_project, ports=ports), indent=2))
    compose("milvus", milvus_project, "up", "-d", "--wait", "--wait-timeout", "300")
    compose("full", project, "build", "api")
    compose("full", project, "run", "--rm", "--no-deps", "api", "python", "-m", "deployment.init_schema")
    compose("full", project, "up", "-d", "--no-build", "--wait", "--wait-timeout", "360", "api", "ui")
    print(f"API http://127.0.0.1:{ports['api']} ; Streamlit http://127.0.0.1:{ports['ui']}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # A subprocess argument must not be logged if this script is extended to use secrets.
        sys.exit(f"Deployment command failed (exit {exc.returncode}). Inspect the project and keep its volumes.")
