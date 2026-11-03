#!/usr/bin/env python3
"""Plan or execute fixed formal runs; no parameter search or foreign job control."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
DATA={'CIFAR10':('torch/CIFAR10',10,'train','test'),'CIFAR100':('torch/CIFAR100',100,'train','test'),
      'STL10':('torch/STL10',10,'train','test'),'Food101':('torch/Food101',101,'train','test'),
      'Flowers102':('torch/Flowers102',102,'trainval','test'),'OxfordIIITPets':('torch/OxfordIIITPet',37,'trainval','test'),
      'ImageNet':('',1000,'train','val')}
METHODS=('default','mimetic','structured','FE')


def tasks(suite):
    rows=[]
    datasets=list(DATA)[:-1] if suite=='small' else ['ImageNet']
    for seed in (210,211,212):
        for ds in datasets:
            models=('Tiny','Small') if ds in ('CIFAR100','Food101','ImageNet') else ('Tiny',)
            for model in models:
                for method in METHODS:
                    rows.append(dict(dataset=ds,model=model,method=method,seed=seed,
                        id=f'{ds}_{model}_{method}_s{seed}',gpus=4 if ds=='ImageNet' else 1))
    return rows


def free_gpus(allowed):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    busy_raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    busy=set(busy_raw.splitlines());free=[]
    for line in raw.splitlines():
        idx,uuid,used=[x.strip() for x in line.split(',')]
        if idx in allowed and uuid not in busy and int(used)<1024:free.append(idx)
    return free


def atomic(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run']);p.add_argument('--suite',choices=['small','imagenet'],required=True)
    p.add_argument('--data-roots',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--acceptance',type=Path)
    p.add_argument('--max-gpu-hours',type=float,help='New authorized total GPUh cap, including already accounted attempts')
    p.add_argument('--diagnostic-manifest',type=Path)
    a=p.parse_args();rows=tasks(a.suite)
    if a.action=='plan':print(json.dumps(rows,indent=2));return
    if not all((a.data_roots,a.output,a.acceptance,a.max_gpu_hours)) or a.max_gpu_hours<=0:
        p.error('run requires data-roots, output, acceptance and a positive authorized GPUh budget')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Clean frozen source required')
    acceptance=json.loads(a.acceptance.read_text());roots=json.loads(a.data_roots.read_text())
    if (acceptance.get('status')!='PASS' or acceptance.get('source_commit')!=head
        or acceptance.get('data_roots_sha256')!=hashlib.sha256(a.data_roots.read_bytes()).hexdigest()):
        raise ValueError('Actual target-host acceptance receipt must bind source and data roots')
    required={'initialization_equivalence','optimizer','full_state_resume','amp','tail_accumulation','data_identity','exact_evaluation'}
    if a.suite=='imagenet':required|={'ddp4','ra_sampler','global_batch1024'}
    if any(acceptance.get('checks',{}).get(k)!='PASS' for k in required):raise ValueError('Incomplete actual runtime acceptance')
    # Estimates must be end-to-end GPUh per full run, not steps/sec alone.
    costs=acceptance['estimated_full_run_gpu_hours']
    if any(float(costs.get(x['dataset']+'/'+x['model'],0))<=0 for x in rows):raise ValueError('Missing measured cost model')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'{a.suite}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=out/f'{a.suite}_ledger.json'
    ledger=json.loads(state_path.read_text()) if state_path.exists() else dict(source_commit=head,gpu_hours=0.,runs={})
    if ledger['source_commit']!=head:raise ValueError('Existing ledger belongs to another source')
    if any(x.get('status')=='running' for x in ledger['runs'].values()):
        raise ValueError('Previous owner has running attempts; reconcile real processes and cost before restart')
    config_dir=out/'resolved_configs';config_dir.mkdir(exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    pending=[]
    for row in rows:
        status=ledger['runs'].get(row['id'],{}).get('status')
        if status=='complete':continue
        if status=='failed':raise ValueError('Failed attempt requires evidence review before retry: '+row['id'])
        if (out/row['id']).exists():raise ValueError('Unowned output directory exists: '+row['id'])
        pending.append(row)
    active=[];failed=False
    while pending or active:
        for item in active[:]:
            proc,row,start,log=item;code=proc.poll()
            if code is None:continue
            elapsed=(time.monotonic()-start)*row['gpus']/3600;ledger['gpu_hours']+=elapsed
            final=out/row['id']/'final_metrics.json'
            complete=code==0 and final.exists()
            ledger['runs'][row['id']].update(status='complete' if complete else 'failed',exit_code=code,gpu_hours=elapsed)
            active.remove(item);log.close();failed|=not complete;atomic(state_path,ledger)
        free=free_gpus(set(a.gpus.split(',')))
        while pending and not failed and len(free)>=pending[0]['gpus']:
            row=pending[0];estimate=float(costs[row['dataset']+'/'+row['model']])
            reserved=sum(max(float(costs[x[1]['dataset']+'/'+x[1]['model']]),(time.monotonic()-x[2])*x[1]['gpus']/3600) for x in active)
            if ledger['gpu_hours']+reserved+estimate>a.max_gpu_hours:failed=True;break
            pending.pop(0);selected=free[:row['gpus']];free=free[row['gpus']:]
            cfg=json.loads((ROOT/'configs'/('imagenet.yaml' if a.suite=='imagenet' else 'small_data.yaml')).read_text())
            ds,classes,train,val=DATA[row['dataset']]
            dim,heads=(192,3) if row['model']=='Tiny' else (384,6)
            cfg.update(dataset=ds,data_dir=roots[row['dataset']],num_classes=classes,train_split=train,val_split=val,
                model_kwargs=dict(img_size=224,weight_init='',embed_dim=dim,depth=12,num_heads=heads),
                init_method=row['method'],seed=row['seed'],output=str(out),experiment=row['id'])
            # Representative snapshots are chosen prospectively, never by score.
            representative=row['seed']==210 and row['dataset'] in ('CIFAR100','ImageNet')
            if representative:
                cfg['snapshot_epochs']=[0,1,5,10,50]
                if a.diagnostic_manifest:
                    manifests=json.loads(a.diagnostic_manifest.read_text())
                    cfg.update(diagnostic_data_manifest=manifests[row['dataset']],diagnostic_samples=16,diagnostic_rank=True)
            path=config_dir/(row['id']+'.yaml');atomic(path,cfg)
            cmd=[sys.executable]
            if row['gpus']>1:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node',str(row['gpus'])]
            cmd += [str(ROOT/'train.py'),'--config',str(path)]
            log=(logs/(row['id']+'.log')).open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(selected))
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.monotonic();active.append((proc,row,start,log))
            ledger['runs'][row['id']]=dict(status='running',pid=proc.pid,gpus=selected,command=cmd,started_utc=time.time())
            atomic(state_path,ledger)
        spent=ledger['gpu_hours']+sum((time.monotonic()-x[2])*x[1]['gpus']/3600 for x in active)
        if spent>a.max_gpu_hours:
            import signal
            for proc,_,_,_ in active:os.killpg(proc.pid,signal.SIGINT)
            failed=True
        if failed and not active:raise SystemExit('Stopped: failure or budget cap; evidence retained')
        if active or pending:time.sleep(10)
    print('All assigned fixed-endpoint runs complete')

if __name__=='__main__':main()
