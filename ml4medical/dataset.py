import torch
import torch.distributed as dist
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl

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
            transforms_list = [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=180, shear=5, fill=250),
                transforms.RandomResizedCrop(224, scale=[0.6, 1], ratio=(3. / 4., 4. / 3.)),
                transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 2)),
                #transforms.RandomAutocontrast(p=0.1),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [transforms.ToTensor()]

        if normalize:
            transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        image_transforms = transforms.Compose(transforms_list)
        super(TileImageDataset, self).__init__(root=os.path.join(root_folder, mode), transform=image_transforms, is_valid_file=is_valid_file)
        self.apply_transforms = True
        if get_parent_slide is None:
            if data_variant == 'CCC':
                self.get_parent_slide = parent_slide_name_CCC
                self.inverse_targets = self.class_to_idx['pos'] != 1 # This is necessary for ROC_AUC metric to work properly
            elif data_variant == 'MSIMSS':
                self.get_parent_slide = parent_slide_name_MSIMSS
                self.inverse_targets = self.class_to_idx['MSIMUT'] != 1 # This is necessary for ROC_AUC metric to work properly
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
            if self.transform is not None and self.apply_transforms:
                sample = self.transform(sample)
            else:
                sample = transforms.ToTensor()(sample)
            if self.target_transform is not None and self.apply_transforms:
                target = self.target_transform(target)
            target = 1 - target if self.inverse_targets else target
            samples.append(sample)
            targets.append(target)
        if no_subbatch:
            samples = samples[0]
            targets = targets[0]
        else:
            samples = torch.stack(samples, dim=0)
            targets = torch.tensor(targets)
            
        return samples, targets
    
    def get_tiles_from_slide(self, slide, tiles=None, cls=None):
        parent_slides = self.parent_slide if cls is None or isinstance(slide, str) else self.parent_slide_per_class[cls]
        slide = list(parent_slides.keys())[slide] if isinstance(slide, int) else slide
        slide_tile_idxs = parent_slides[slide]
        
        if tiles and isinstance(tiles, int):
            # pick n random tiles from slide:
            tiles = torch.randint(len(slide_tile_idxs), tiles)
            slide_tile_idxs = slide_tile_idxs[tiles]
        
        samples, targets = self[slide_tile_idxs]
        
        if self.tile_position:
            pos = [self.tile_position[idx] for idx in slide_tile_idxs]
            return samples, targets, pos
        
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
                # shuffle wihtin classes. This is to make sure we oversample/drop different entries each epoch
                sample_idxs = [idxs[torch.randperm(len(idxs), generator=g)] for idxs in sample_idxs]
            sample_idxs = [idxs[:sample_len] for idxs in sample_idxs]

            idxs_in_subbatches_per_class = [idxs_in_subbatches[idxs, :]
                                            for idxs_in_subbatches, idxs in zip(idxs_in_subbatches_per_class, sample_idxs)]
            
        
        idxs_in_subbatches = torch.cat(idxs_in_subbatches_per_class, 0) #merge indices from classes into one
        if self.shuffle:
            # we need to shuffle once more after merging the indices of the classes!
            idxs_in_subbatches = idxs_in_subbatches[torch.randperm(idxs_in_subbatches.shape[0], generator=g), :]
            
        idxs_in_subbatches = idxs_in_subbatches[:self.total_size, :]
        idxs_in_subbatches = idxs_in_subbatches[self.rank:self.total_size:self.num_replicas, 0 if self.no_subbatches else ...].tolist()
        return iter(idxs_in_subbatches)
        
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    
class DataModule(pl.LightningDataModule):
        def __init__(self,
                     root_folder,
                     data_variant,
                     train_batch_size,
                     val_batch_size,
                     train_subbatch_size,
                     val_subbatch_size,
                     train_subbatch_mode='class',
                     val_subbatch_mode='slide',
                     train_balance=None,
                     val_balance=None,
                     distributed=False,
                     num_workers=1,
                     persistent_workers=False,
                     ):
            super(DataModule, self).__init__()
            normalize = data_variant != 'MSIMSS'
            self.train_dataset = TileImageDataset(root_folder=root_folder, mode='train', data_variant=data_variant, normalize=normalize)
            self.val_dataset = TileImageDataset(root_folder=root_folder, mode='val', data_variant=data_variant, normalize=normalize)
            self.train_batch_size = train_batch_size
            self.val_batch_size = val_batch_size
            self.train_subbatch_size = train_subbatch_size
            self.val_subbatch_size = val_subbatch_size
            self.train_subbatch_mode = train_subbatch_mode
            self.val_subbatch_mode = val_subbatch_mode
            self.train_balance = train_balance
            self.val_balance = val_balance
            self.distributed = distributed
            self.num_workers = num_workers
            self.persistent_workers = persistent_workers
            
        def train_dataloader(self):
            sampler = TileSubBatchSampler(self.train_dataset, subbatch_size=self.train_subbatch_size, mode=self.train_subbatch_mode,
                                          shuffle=True, balance=self.train_balance, distributed=self.distributed)
            return DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=sampler, num_workers=self.num_workers,
                              persistent_workers=self.persistent_workers)
        
        def val_dataloader(self):
            sampler = TileSubBatchSampler(self.val_dataset, subbatch_size=self.val_subbatch_size, mode=self.val_subbatch_mode,
                                          shuffle=False, balance=self.val_balance, distributed=self.distributed)
            return DataLoader(self.val_dataset, batch_size=self.val_batch_size, sampler=sampler, num_workers=self.num_workers,
                              persistent_workers=self.persistent_workers)