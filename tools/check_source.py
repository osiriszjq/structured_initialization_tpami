#!/usr/bin/env python3
"""No training dependencies: check syntax and the documented timm delta."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    upstream=json.loads((ROOT/'docs/upstream_blobs.json').read_text())
    changed=[];added=[]
    for p in (ROOT/'timm').rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(ROOT).as_posix();b=p.read_bytes()
        sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if rel not in upstream:added.append(rel)
        elif sha!=upstream[rel]:changed.append(rel)
    assert changed==['timm/data/dataset_factory.py'],changed
    assert set(added)=={'timm/models/structured_init.py','timm/utils/structured_runtime.py',
        'timm/utils/structured_diagnostics.py','timm/data/structured_datasets.py'},added
    count=0
    for p in ROOT.rglob('*.py'):
        if '.git' not in p.parts:ast.parse(p.read_text(),filename=str(p));count+=1
    assert len(list((ROOT/'timm/data').rglob('*.py')))==35
    print(json.dumps(dict(status='PASS_STATIC_ONLY',changed_timm=changed,added_timm=added,python_files=count)))
if __name__=='__main__':main()
