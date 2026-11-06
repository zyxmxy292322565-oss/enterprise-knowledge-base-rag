"""Location-independent receipt for model files already verified by the caller."""
import json
from hashlib import sha256
from pathlib import Path


def model_receipt(name, root, verified_rows):
    root = Path(root)
    files = sorted((dict(path=Path(row['path']).relative_to(root).as_posix(), sha256=row['sha256'])
                    for row in verified_rows), key=lambda row: row['path'])
    weights = [row for row in files if row['path'] in {'model.safetensors', 'pytorch_model.bin'}]
    if len(weights) != 1:
        raise ValueError('Missing or ambiguous model weights fingerprint')
    return dict(name=name, revision=sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                weights_sha256=weights[0]['sha256'], location=str(root), files=files)
