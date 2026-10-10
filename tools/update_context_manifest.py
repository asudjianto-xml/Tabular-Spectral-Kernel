"""Refresh distributed hashes after an intentional source-only portability edit."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parents[1] / 'src/tabular_spectral_kernel/context_bundle'
p = root / 'release_manifest.json'
manifest = json.loads(p.read_text())
for file in sorted(root.rglob('*')):
    if not file.is_file() or file == p or '__pycache__' in file.parts:
        continue
    key = file.relative_to(root).as_posix()
    entry = manifest.setdefault(key, {})
    entry['sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
p.write_text(json.dumps(manifest, indent=2) + '\n')
