#!/usr/bin/env python3
"""Optional remote equivalence test against a preserved scientific source tree.

The reference tree and accepted config are private inputs, never bundled into
this public repository. This test opens no images and launches no training.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Fresh output required')
import torch
import timm
from timm.models.structured_init import initialize,isolated_cpu_rng,tensor_digest
assert Path(timm.__file__).resolve().is_relative_to(ROOT)
package=a.reference_root/'transition_lab'
spec=importlib.util.spec_from_file_location('reference_transition_lab',package/'__init__.py',submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from reference_transition_lab.pelearn import initialize as reference_initialize
from reference_transition_lab.joint_protocol import fingerprint
from reference_transition_lab.joint_pe_entropy import source_identity
cfg=json.loads(a.config.read_text());assert cfg['pe_alpha']==18. and cfg['joint_entropy_target']==.90
a.output.mkdir(parents=True);torch.set_num_threads(2)
def base():
    with isolated_cpu_rng(cfg['seed']):
        return timm.create_model(cfg['model'],pretrained=False,in_chans=3,num_classes=cfg['num_classes'],
            global_pool=cfg['gp'],drop_rate=cfg['drop'],drop_path_rate=cfg['drop_path'],**cfg['model_kwargs'])
rows=[]
for method in ('F','L','FE','LE'):
    c=dict(cfg,pe_representation=method[0],joint_entropy_on=method in ('FE','LE'),joint_require_prepared=False,
        joint_source_commit=source_identity()['commit'],pe_root=str(a.output/'reference_cache'))
    c['joint_recipe_sha256']=fingerprint(c)
    ref=base();reference_initialize(ref,SimpleNamespace(**c))
    candidate=base();initialize(candidate,method,cfg['seed'])
    assert ref.state_dict().keys()==candidate.state_dict().keys()
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in ref.state_dict().items())
    rows.append(dict(method=method,state_sha256=tensor_digest(candidate.state_dict())))
(a.output/'equivalence.json').write_text(json.dumps(dict(status='PASS_BITWISE_REFERENCE_EQUIVALENCE',rows=rows),indent=2)+'\n')
