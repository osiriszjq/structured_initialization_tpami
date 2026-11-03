"""Explicit official classification splits for four small-data benchmarks."""
from torch.utils.data import ConcatDataset
from torchvision.datasets import Food101, STL10, Flowers102, OxfordIIITPet


class FlowersTrainVal(ConcatDataset):
    # timm create_loader assigns dataset.transform; forward it to both parts.
    @property
    def transform(self): return self.datasets[0].transform

    @transform.setter
    def transform(self, value):
        for dataset in self.datasets: dataset.transform=value


def classification_dataset(name, root, split, download=False):
    common=dict(root=root,download=download)
    if name=='food101' and split in ('train','test'):
        return Food101(split=split,**common)
    if name=='stl10' and split in ('train','test'):
        return STL10(split=split,folds=None,**common)
    if name=='flowers102':
        if split=='trainval':return FlowersTrainVal([Flowers102(split=s,**common) for s in ('train','val')])
        if split=='test':return Flowers102(split='test',**common)
    if name=='oxfordiiitpet' and split in ('trainval','test'):
        return OxfordIIITPet(split=split,target_types='category',**common)
    raise ValueError('Explicit official split required; no unlabeled data or hidden holdout')
