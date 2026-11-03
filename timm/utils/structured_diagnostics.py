"""Read-only attention diagnostics on fixed TRAIN image IDs, with no fitting."""
import copy
import math
import random
import numpy as np
import torch
from .structured_runtime import atomic_json,root


@torch.no_grad()
def measure(model,eval_loader,args,completed):
    if not args.diagnostic_data_manifest: raise ValueError('Diagnostics require a frozen TRAIN image manifest')
    import json
    from pathlib import Path
    from PIL import Image
    spec=json.loads(Path(args.diagnostic_data_manifest).read_text())
    if spec.get('partition')!='train':raise ValueError('Diagnostic samples must be from TRAIN')
    entries=spec['samples'][:args.diagnostic_samples]
    if not entries:raise ValueError('No diagnostic samples')
    # Restore all RNG and train/eval state so diagnostics cannot change the recipe.
    py=random.getstate();npstate=np.random.get_state();was_training=model.training
    records=[];hooks=[]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        try:
            model.eval()
            layers=args.diagnostic_layers or [0,len(model.blocks)//2,len(model.blocks)-1]
            def hook(layer):
                def observe(module,inputs):
                    x=inputs[0];b,n,d=x.shape
                    q,k,_=module.qkv(x).reshape(b,n,3,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
                    z=(q*module.scale)@k.transpose(-1,-2)
                    p=z.float().softmax(-1)[:,:,1:,:]
                    patch=p[:,:,:,1:];patch=patch/patch.sum(-1,keepdim=True)
                    row=dict(layer=layer,entropy=(-(p*p.clamp_min(1e-30).log()).sum(-1)/math.log(n)).mean((0,2)).cpu().tolist(),
                        cls_mass=p[:,:,:,0].mean((0,2)).cpu().tolist())
                    if args.diagnostic_rank:
                        sv=torch.linalg.svdvals(patch.cpu().double());prob=sv/sv.sum(-1,keepdim=True)
                        row.update(effective_rank=(-(prob*prob.clamp_min(1e-300).log()).sum(-1)).exp().mean(0).tolist(),
                            stable_rank=(sv.square().sum(-1)/sv[...,0].square()).mean(0).tolist(),
                            singular_values=sv.tolist())
                    records.append(row)
                return observe
            for layer in layers:hooks.append(model.blocks[layer].attn.register_forward_pre_hook(hook(layer)))
            from timm.data import create_transform
            transform=create_transform(input_size=args.input_size,is_training=False,mean=args.mean,std=args.std,
                interpolation=args.interpolation,crop_pct=args.crop_pct)
            # One image at a time bounds attention/SVD memory and preserves exact IDs.
            import hashlib
            for entry in entries:
                p=Path(spec['root'])/entry['path']
                if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Diagnostic image changed')
                with Image.open(p) as im:x=transform(im.convert('RGB')).unsqueeze(0)
                model(x.to(next(model.parameters()).device))
            atomic_json(root(args)/f'diagnostics_epoch{completed:03d}.json',dict(samples=entries,records=records,
                semantics='Full keys incl CLS for entropy; row-renormalized patch block for rank. FP64 SVD; no fitting.'))
        finally:
            for handle in hooks:handle.remove()
            model.train(was_training);random.setstate(py);np.random.set_state(npstate)
