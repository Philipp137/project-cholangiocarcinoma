import torch
import torch.distributed as dist
import numpy as np
import os
from torchvision import transforms, datasets
from pathlib import Path

def is_valid_file(fpath):
    return fpath.lower().endswith('.png')
    
    
def parent_slide_name_MSIMSS(tile_path):
    _, tile_name = os.path.split(tile_path)
    return '-'.join(tile_name.split('-')[2:5])


def parent_slide_name_CCC(tile_path):
    return os.path.split(Path(tile_path).resolve().parents[1])[-1]


def tile_position_CCC(tile_path):
    _, tile_name = os.path.split(tile_path)
    tile_name_component = tile_name.split('_')
    idx_i = int(tile_name_component[-2])
    idx_j = int(tile_name_component[-1].replace('.png', ''))
    return idx_i, idx_j
    
    
class TileImageDataset(datasets.ImageFolder):
    def __init__(self, root_folder, mode, normalize=True, data_variant='CCC', get_parent_slide=None, get_tile_position=None):
        #TODO: Move Transforms out?
        if mode == 'train':
            transofrms_list = [
                transforms.RandomResizedCrop(224, scale=[0.8, 1], ratio=[5. / 6., 6. / 5.]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transofrms_list = [transforms.ToTensor()]

        if normalize:
            transofrms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        image_transforms = transforms.Compose(transofrms_list)
        super(TileImageDataset, self).__init__(root=os.path.join(root_folder, mode), transform=image_transforms, is_valid_file=is_valid_file)
        
        if get_parent_slide is None:
            if data_variant == 'CCC':
                self.get_parent_slide = parent_slide_name_CCC
            elif data_variant == 'MSIMSS':
                self.get_parent_slide = parent_slide_name_MSIMSS
        else:
            self.get_parent_slide = get_parent_slide
            
        if get_tile_position is None:
            if data_variant == 'CCC':
                self.get_tile_position = tile_position_CCC
            elif data_variant == 'MSIMSS':
                self.get_tile_position = None
        else:
            self.get_tile_position = get_tile_position
            
        self.get_tile_position = get_tile_position
        self.parent_slide = dict()
        self.parent_slide_name = list()
        self.parent_slide_class =list()
        self.tile_position = list()
        self.idx_in_class = {n: [] for n in range(len(self.classes))}
        self._find_tile_info()
        
    def _find_tile_info(self):
        for idx, (pth, cls) in enumerate(self.samples):
            if self.get_parent_slide is not None:
                parent_slide = self.get_parent_slide(pth)
                if parent_slide not in self.parent_slide:
                    self.parent_slide[parent_slide] = [idx]
                    self.parent_slide_name.append(parent_slide)
                    self.parent_slide_class.append(cls)
                else:
                    self.parent_slide[parent_slide].append(idx)
            
            if self.get_tile_position is not None:
                self.tile_position.append(self.get_tile_position(pth))
            self.idx_in_class[cls].append(idx)
        #self.tile_position = np.array(self.tile_position) if self.tile_position else self.tile_position
        
    def __getitem__(self, index):
        no_subbatch = False
        if not hasattr(index, '__len__'):
            index = [index]
            no_subbatch = True
        samples = []
        targets = []
        for idx in index:
            path, target = self.samples[idx]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            samples.append(sample)
            targets.append(target)
        if no_subbatch:
            samples = samples[0]
            targets = targets[0]
        else:
            samples = torch.stack(samples, dim=0)
            targets = torch.tensor(targets)
        return samples, targets
        
        
class TileSubBatchSampler(torch.utils.data.Sampler):
    def __init__(self, tile_image_dataset, subbatch_size, mode='class', shuffle=True, shuffle_subs=None, distributed=False,
                 balance='oversample'):
        self.subbatch_size = subbatch_size if subbatch_size > 0 else 1
        self.no_subbatches = subbatch_size == 0
        self.ds = tile_image_dataset
        self.mode = mode
        self.shuffle = shuffle
        self.shuffle_subs = shuffle if shuffle_subs is None else shuffle_subs
        self.distributed = distributed
        self.balance = balance
        self.n_samples_per_class = [len(cls_idx) for cls_idx in self.ds.idx_in_class.values()]
        self.n_classes = len(self.ds.idx_in_class)
        if mode == 'slide':
            self.subs_idx = [torch.tensor(idx) for idx in self.ds.parent_slide.values()]
        elif mode == 'class':
            self.subs_idx = [torch.tensor(idx) for idx in self.ds.idx_in_class.values()]
        self.class_sample_diffs = [max(self.n_samples_per_class) - n_samples for n_samples in self.n_samples_per_class]

        if balance == 'oversample':
            self.n_subbatches = (max(self.n_samples_per_class) // self.subbatch_size) * self.n_classes
        elif balance == 'undersample':
            self.n_subbatches = (min(self.n_samples_per_class) // self.subbatch_size) * self.n_classes
        else:
            self.n_subbatches = sum([len(sub_idx) // self.subbatch_size for sub_idx in self.subs_idx])
        
        # taken/adapted from torch.distributed.DistributedSampler:
        if distributed:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

            self.num_replicas = num_replicas
            self.rank = rank
            self.total_size = self.n_subbatches - (self.n_subbatches % self.num_replicas)
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_replicas = None
            self.rank = None
            self.total_size = self.n_subbatches
            self.num_samples = self.n_subbatches
            

        self.epoch = 0
        
    
    def get_subbatched_sample_indexes(self):
        idx_per_sub = [sub_idx[torch.randperm(len(sub_idx))] for sub_idx in self.subs_idx] if self.shuffle_subs else self.subs_idx
        
        if self.mode == 'class':
            # In case of self.shuffle_subs == False, this is not very clean, since for each epoch we either always oversample the same first
            # samples in each sub or drop the same last samples in each sub when undersampling. But in case of self.shuffle_subs == True,
            # we randomly drop or oversample different samples in each epoch.
            if self.balance == 'undersample':
                idx_per_sub = [sub_idx[:min(self.n_samples_per_class)] for sub_idx in idx_per_sub]
            elif self.balance == 'oversample':
                idx_per_sub = [torch.cat([sub_idx, sub_idx[:self.class_sample_diffs[n]]], 0) for n, sub_idx in enumerate(idx_per_sub)]
            
        idxs_in_subbatches = torch.cat([
            torch.cat([sub_idx[i:i + self.subbatch_size][None, :] for i in range(0, len(sub_idx), self.subbatch_size)
                       if len(sub_idx[i:i + self.subbatch_size]) == self.subbatch_size], dim=0)
            for sub_idx in idx_per_sub if len(sub_idx) > self.subbatch_size], dim=0)
        
        return idxs_in_subbatches
    
    def __iter__(self):
        idxs_in_subbatches = self.get_subbatched_sample_indexes()
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(idxs_in_subbatches), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(idxs_in_subbatches)))
        indices = indices[:self.total_size]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        idx_2 = 0 if self.no_subbatches else ...
        subbatched_idx = idxs_in_subbatches[indices, idx_2].tolist()
        return iter(subbatched_idx)
        
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
