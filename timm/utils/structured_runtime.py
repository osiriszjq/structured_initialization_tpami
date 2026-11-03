"""Small train.py hooks for provenance, exact evaluation and bounded snapshots."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist
from timm.utils.model import unwrap_model


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n');temp.replace(path)


def root(args): return Path(args.output)/args.experiment


def recipe(args):
    operational={'output','experiment','resume','start_epoch','stop_after_epochs','rank','local_rank',
        'device','device_modules','distributed','log_interval','log_wandb','wandb_project','wandb_tags',
        'wandb_resume_id','snapshot_epochs','diagnostic_samples','diagnostic_layers','diagnostic_rank',
        'dist_url','dist_backend','diagnostic_data_manifest'}
    return {k:v for k,v in vars(args).items() if k not in operational}


def fingerprint(args):
    return hashlib.sha256(json.dumps(recipe(args),sort_keys=True,default=str).encode()).hexdigest()


def setup(model,args):
    from timm.models.structured_init import initialize,restore_topology
    if not args.init_method: return None
    if not args.output or not args.experiment: raise ValueError('Explicit --output and --experiment required')
    if (args.pretrained or args.pretrained_path or args.initial_checkpoint or args.model_ema
        or args.torchscript or args.torchcompile or args.aug_splits or args.use_multi_epochs_loader
        or args.naflex_loader or args.recovery_interval or args.no_resume_opt):
        raise ValueError('Formal path requires fresh plain ViT, no EMA/compile/mid-epoch recovery')
    if args.model_kwargs.get('weight_init','')!='': raise ValueError('Formal comparison uses native timm base')
    if not args.no_prefetcher: raise ValueError('Use --no-prefetcher for auditable data/diagnostics')
    if args.workers<0 or args.layer_decay!=1.: raise ValueError('Formal optimizer requires layer_decay=1')
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo): raise ValueError('Freeze clean code before running')
    error=None
    if args.rank==0:
        if root(args).exists() and not args.resume: error='Fresh output directory already exists'
    if args.distributed:
        payload=[error];dist.broadcast_object_list(payload,src=0);error=payload[0]
    if error: raise ValueError(error)
    root(args).mkdir(parents=True,exist_ok=True)
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('source_commit')!=commit or saved.get('recipe_sha256')!=fingerprint(args):
            raise ValueError('Resume source/recipe mismatch')
        restore_topology(model,args.init_method)
        report=dict(resume=str(Path(args.resume).resolve()),calibration_executed=False)
    else:
        report=initialize(model,args.init_method,args.seed)
    if args.rank==0:
        atomic_json(root(args)/('resume_initialization.json' if args.resume else 'initialization.json'),report)
        atomic_json(root(args)/('resume_manifest.json' if args.resume else 'run_manifest.json'),dict(source_commit=commit,recipe_sha256=fingerprint(args),
            resolved_args=vars(args),python=__import__('sys').version,torch=torch.__version__,
            torchvision=__import__('torchvision').__version__,cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),started_utc=time.time()))
        if not args.resume and 0 in args.snapshot_epochs:
            _weights(model,root(args)/'weights_epoch000.pth',args,0)
    return report


def _weights(model,path,args,epoch):
    from timm.models.structured_init import folded
    ordinary=folded(unwrap_model(model))
    payload=dict(state_dict={k:v.detach().cpu() for k,v in ordinary.state_dict().items()},
        init_method=args.init_method,completed_epochs=epoch,model=args.model,model_kwargs=args.model_kwargs,
        num_classes=args.num_classes,recipe_sha256=fingerprint(args))
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def exact_validate(validate,model,loader,loss_fn,args,**kwargs):
    """Rank0 evaluates the complete unpadded set; broadcast its metrics."""
    if not args.init_method: return validate(model,loader,loss_fn,args,**kwargs)
    value=None
    if args.rank==0:
        local=copy.copy(args);local.distributed=False
        value=validate(unwrap_model(model),loader,loss_fn,local,**kwargs)
    if args.distributed:
        payload=[value];dist.broadcast_object_list(payload,src=0);value=payload[0]
    return value


def resume_schedule(scheduler,optimizer,args):
    if args.resume and args.init_method:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        scheduler.load_state_dict(state['scheduler'])
        for group,saved in zip(optimizer.param_groups,state['optimizer']['param_groups']):
            group['lr']=saved['lr']
        # Optimizer/scaler/model already restored by timm's resume_checkpoint.
        states=state['rng_by_rank']
        if len(states)!=args.world_size: raise ValueError('Resume world size changed')
        rng=states[args.rank]
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])


def epoch_seed(args,epoch):
    # Nonpersistent workers are recreated from this deterministic epoch seed.
    from timm.utils import random_seed
    random_seed(args.seed+100000+epoch*1009,args.rank)


def dataset_identity(train,evaluation,args):
    if args.rank!=0:return
    def info(ds):
        from torch.utils.data import ConcatDataset
        if isinstance(ds,ConcatDataset):return dict(count=len(ds),parts=[info(x) for x in ds.datasets])
        labels=getattr(ds,'targets',getattr(ds,'labels',getattr(ds,'_labels',None)))
        if labels is None and hasattr(ds,'samples'): labels=[v[1] for v in ds.samples]
        classes=getattr(ds,'class_to_idx',getattr(ds,'classes',None))
        label_sha=hashlib.sha256(np.asarray(labels).tobytes()).hexdigest() if labels is not None else None
        return dict(type=type(ds).__name__,count=len(ds),labels_sha256=label_sha,classes=classes)
    atomic_json(root(args)/'data_identity.json',dict(train=info(train),evaluation=info(evaluation),
        data_root=str(Path(args.data_dir or args.data).resolve()),train_split=args.train_split,val_split=args.val_split,
        note='Label/count identity only; receiving agent must separately hash authoritative metadata and confirm legal origin.'))


def epoch_end(model,optimizer,scheduler,scaler,args,epoch,num_epochs,train_metrics,eval_metrics,loader_eval):
    if not args.init_method:return
    m=unwrap_model(model)
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
             cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    states=[None]*args.world_size
    if args.distributed:dist.all_gather_object(states,rng)
    else:states=[rng]
    if args.rank==0:
        start=time.monotonic()
        from timm.models.structured_init import tensor_digest
        repo=Path(__file__).resolve().parents[2]
        payload=dict(epoch=epoch,version=2,arch=args.model,state_dict=m.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng_by_rank=states,recipe_sha256=fingerprint(args),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
        if scaler is not None:payload[scaler.state_dict_key]=scaler.state_dict()
        temp=root(args)/'latest.tmp.pth';latest=root(args)/'latest.pth';previous=root(args)/'previous.pth'
        torch.save(payload,temp)
        if latest.exists():latest.replace(previous)
        temp.replace(latest)
        completed=epoch+1
        if completed in args.snapshot_epochs:
            _weights(model,root(args)/f'weights_epoch{completed:03d}.pth',args,completed)
        if completed in args.snapshot_epochs or completed==num_epochs:
            if args.diagnostic_samples:
                from .structured_diagnostics import measure
                measure(m,loader_eval,args,completed)
        final=dict(epoch=epoch,train=dict(train_metrics),evaluation=dict(eval_metrics or {}),
            model_state_sha256=tensor_digest(m.state_dict()),checkpoint_seconds=time.monotonic()-start)
        atomic_json(root(args)/'last_metrics.json',final)
        if completed==num_epochs:
            _weights(model,root(args)/'final_model.pth',args,completed)
            final['final_model_sha256']=hashlib.sha256((root(args)/'final_model.pth').read_bytes()).hexdigest()
            atomic_json(root(args)/'final_metrics.json',final)
            # Keep full final state only for prospectively chosen representative runs.
            if not args.snapshot_epochs:
                latest.unlink()
                if previous.exists():previous.unlink()
    if args.distributed:dist.barrier()


def accumulation_window(index, batches, full):
    if not 0<=index<batches or full<1: raise ValueError('invalid accumulation window')
    start=(index//full)*full
    size=min(full,batches-start)
    return size,index==start+size-1,index//full


class ObservedScaler:
    """Delegate numerical AMP unchanged; record attempted and skipped updates."""
    def __init__(self,inner):
        self.inner=inner;self.state_dict_key=inner.state_dict_key
        self.attempted=0;self.skipped=0
    def __call__(self,*args,**kwargs):
        before=self.inner._scaler.get_scale()
        result=self.inner(*args,**kwargs)
        if kwargs.get('need_update',True):
            self.attempted+=1
            self.skipped+=int(self.inner._scaler.get_scale()<before)
        return result
    def state_dict(self):return self.inner.state_dict()
    def load_state_dict(self,value):return self.inner.load_state_dict(value)
