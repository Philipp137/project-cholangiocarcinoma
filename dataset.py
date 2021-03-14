import torch
import torch.distributed as dist
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
        self.parent_slide_per_class = {n: dict() for n in range(len(self.classes))}
        self.parent_slide_name = list()
        self.parent_slide_class = list()
        self.tile_position = list()
        self.idx_in_class = {n: [] for n in range(len(self.classes))}
        self._find_tile_info()
        
    def _find_tile_info(self):
        for idx, (pth, cls) in enumerate(self.samples):
            if self.get_parent_slide is not None:
                parent_slide = self.get_parent_slide(pth)
                if parent_slide not in self.parent_slide:
                    self.parent_slide[parent_slide] = [idx]
                    self.parent_slide_per_class[cls][parent_slide] = [idx]
                    self.parent_slide_name.append(parent_slide)
                    self.parent_slide_class.append(cls)
                else:
                    self.parent_slide[parent_slide].append(idx)
                    self.parent_slide_per_class[cls][parent_slide].append(idx)
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
            #targets = torch.tensor(targets)
            targets = targets[0]
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
        self.n_classes = len(self.ds.idx_in_class)
        
        if mode == 'slide':
            self.subs_idx = [[torch.tensor(idx) for idx in subs_idx.values()] for subs_idx in self.ds.parent_slide_per_class.values()]
        elif mode == 'class':
            self.subs_idx = [[torch.tensor(idx)] for idx in self.ds.idx_in_class.values()]
            
        self.n_subbatches_per_class = [sum([len(sub_idx) // self.subbatch_size for sub_idx in subs_in_class]) for subs_in_class in
                                       self.subs_idx]
        if balance == 'oversample':
            self.n_subbatches = max(self.n_subbatches_per_class) * self.n_classes
        elif balance == 'undersample':
            self.n_subbatches = min(self.n_subbatches_per_class) * self.n_classes
        else:
            self.n_subbatches = sum(self.n_subbatches_per_class)
        
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
        
    
    def __iter__(self):
        
        # parts taken/adapted from torch.distributed.DistributedSampler
        
        if self.shuffle or self.shuffle_subs:
            g = torch.Generator()
            g.manual_seed(self.epoch)

        idx_per_sub_per_class = [[sub_idx[torch.randperm(len(sub_idx), generator=g)] for sub_idx in subs_idx_in_class]
                                 for subs_idx_in_class in self.subs_idx] if self.shuffle_subs else self.subs_idx

        idxs_in_subbatches_per_class = [
            torch.cat([
                torch.cat([
                    sub_idx[i:i + self.subbatch_size][None, :] for i in range(0, len(sub_idx), self.subbatch_size)
                    if len(sub_idx[i:i + self.subbatch_size]) == self.subbatch_size
                ], dim=0) for sub_idx in subs_idx_in_class if len(sub_idx) > self.subbatch_size
            ], dim=0) for subs_idx_in_class in idx_per_sub_per_class
        ]
        
        
        if self.shuffle:
            idxs_in_subbatches_per_class = [idxs_in_subbatches[torch.randperm(len(idxs_in_subbatches), generator=g),:]
                                            for idxs_in_subbatches in idxs_in_subbatches_per_class]
            
        # balance classes:
        n_subs_per_class = [len(idxs) for idxs in idxs_in_subbatches_per_class]
        if self.balance and max(n_subs_per_class) > min(n_subs_per_class):
            # In case of self.shuffle == False, this is not very clean, since for each epoch we either always oversample the same first
            # samples in each sub or drop the same last samples in each sub when undersampling. But in case of self.shuffle == True,
            # we randomly drop or oversample different samples in each epoch.

            if self.balance == 'undersample':
                sample_len = min(n_subs_per_class)
            elif self.balance == 'oversample':
                sample_len = max(n_subs_per_class)
            n_runs = torch.ceil(torch.tensor([sample_len/n_subs for n_subs in n_subs_per_class]))
            sample_idxs = [(torch.arange(len_sub*n) % len_sub).long() for len_sub, n in zip(n_subs_per_class, n_runs)]
            if self.shuffle:
                sample_idxs = [idxs[torch.randperm(len(idxs), generator=g)] for idxs in sample_idxs]
            sample_idxs = [idxs[:sample_len] for idxs in sample_idxs]

            idxs_in_subbatches_per_class = [idxs_in_subbatches[idxs, :]
                                            for idxs_in_subbatches, idxs in zip(idxs_in_subbatches_per_class, sample_idxs)]
            
        
        idxs_in_subbatches = torch.cat(idxs_in_subbatches_per_class, 0) #merge indices from classes into one
        if self.shuffle:
            # we need to shuffle once more after merging the indices of the classes!
            idxs_in_subbatches = idxs_in_subbatches[torch.randperm(idxs_in_subbatches.shape[0], generator=g), :]
            
        idxs_in_subbatches = idxs_in_subbatches[:self.total_size, :]
        idxs_in_subbatches = idxs_in_subbatches[self.rank:self.total_size:self.num_replicas, , 0 if self.no_subbatches else ...].tolist()
        return iter(idxs_in_subbatches)
        
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
