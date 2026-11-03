"""Remote CPU numerical tests; run with python -m unittest discover -s tests."""
import copy
import unittest
from types import SimpleNamespace
import torch
from timm import create_model
from timm.models.structured_init import (initialize,restore_topology,isolated_cpu_rng,
    components,optimizer_groups,folded,position_summary,calibrate,check_pair,tensor_digest)


def base():
    with isolated_cpu_rng(210):
        return create_model('vit_tiny_patch16_224',pretrained=False,num_classes=10,
                            depth=1,img_size=224,weight_init='')


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_fe_components_and_calibration(self):
        original=base();initialize(original,'structured',210)
        f=base();initialize(f,'F',210)
        fe=base();initialize(fe,'FE',210)
        b,a=components(fe)
        self.assertFalse(a.requires_grad);self.assertTrue(b.requires_grad);self.assertEqual(float(a),18.)
        torch.testing.assert_close(original.pos_embed,f.pos_embed,atol=0,rtol=0)
        # Folding removes only the storage representation.
        torch.testing.assert_close(folded(f).blocks[0].attn.qkv.weight,original.blocks[0].attn.qkv.weight,atol=0,rtol=0)
        ref=folded(f);candidate=folded(fe);check_pair(ref,candidate)
        self.assertTrue(all(abs(x['normalized_full_entropy_mean']-.9)<3e-5 for x in position_summary(fe)))
        with self.assertRaises(ValueError):initialize(fe,'FE',210)

    def test_optimizer_fold_and_continuation(self):
        x=torch.randn(1,3,224,224)
        for method in ('default','mimetic','structured','F','L','FE','LE'):
            m=base();initialize(m,method,210);m.eval()
            groups=optimizer_groups(m,SimpleNamespace(weight_decay=.05,layer_decay=1.))
            opt=torch.optim.AdamW(groups,lr=.002)
            assigned=[p for g in opt.param_groups for p in g['params']]
            self.assertEqual(len(assigned),len({id(p) for p in assigned}))
            self.assertEqual({id(p) for p in assigned},{id(p) for p in m.parameters()})
            b,a=components(m)
            self.assertEqual(a.requires_grad,method in ('L','LE'))
            for g in opt.param_groups:
                if any(p is b or p is a for p in g['params']):self.assertEqual(g['weight_decay'],0.)
            def step(model,optimizer):
                optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
            step(m,opt)
            state=copy.deepcopy(m.state_dict());os=copy.deepcopy(opt.state_dict())
            restored=base();restore_topology(restored,method);restored.load_state_dict(state);restored.eval()
            other=torch.optim.AdamW(optimizer_groups(restored,SimpleNamespace(weight_decay=.05,layer_decay=1.)),lr=.002)
            other.load_state_dict(os)
            step(m,opt);step(restored,other)
            self.assertEqual(tensor_digest(m.state_dict()),tensor_digest(restored.state_dict()))
            with torch.no_grad():torch.testing.assert_close(m(x),folded(m)(x),atol=1e-6,rtol=1e-6)

    def test_tail_windows(self):
        from timm.utils.structured_runtime import accumulation_window
        for batches in (1,3,8,9,10,15,17):
            windows=[accumulation_window(i,batches,8) for i in range(batches)]
            self.assertEqual(sum(int(v[1]) for v in windows),(batches+7)//8)
            for size,_,idx in windows:self.assertEqual(size,min(8,batches-idx*8))
