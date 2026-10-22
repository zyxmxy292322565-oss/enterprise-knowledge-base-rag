"""Create an explicit, immutable document snapshot inside evaluation/ only."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def create_snapshot(source, snapshot_id):
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError('source must be a directory')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', snapshot_id):
        raise ValueError('invalid snapshot identifier')
    project_cache = (ROOT.parent / 'cache').resolve()
    if source == project_cache or project_cache in source.parents:
        raise ValueError('index/cache cannot be used as document source')
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.txt', '.pdf'})
    if not files or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('source must contain ordinary TXT or PDF documents')
    base = (ROOT / 'snapshots' / 'documents').resolve()
    if ROOT not in base.parents:
        raise ValueError('snapshot directory escapes evaluation root')
    destination = base / snapshot_id
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for p in files:
        content = p.read_bytes()
        (destination / p.name).write_bytes(content)
        entries.append({'name': p.name, 'bytes': len(content),
                        'sha256': hashlib.sha256(content).hexdigest(),
                        'md5': hashlib.md5(content).hexdigest()})
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode('utf-8')
    manifest = {'snapshot_id': snapshot_id, 'files': entries,
                'manifest_sha256': hashlib.sha256(canonical).hexdigest()}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('snapshot_id')
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.source, args.snapshot_id), ensure_ascii=False))
