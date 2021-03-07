from torchvision import transforms, datasets
from pathlib import Path
import numpy as np
import os


def is_valid_file(fpath):
    return fpath.lower().endswith('.png')
    
    
def parent_slide_name_MSSMSI(tile_path):
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
    def __init__(self, root_folder, mode, normalize=True, get_parent_slide=None, get_tile_position=None):
        self.get_parent_slide = get_parent_slide
        self.get_tile_position = get_tile_position
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

        self.parent_slide = dict()
        self.parent_slide_name = list()
        self.tile_position = list()
        self.idx_in_class = {n: np.array([], dtype=int) for n in range(len(self.classes))}
        self._find_tile_info()
        
    def _find_tile_info(self):
        for idx, (pth, cls) in enumerate(self.samples):
            if self.get_parent_slide is not None:
                parent_slide = self.get_parent_slide(pth)
                if parent_slide not in self.parent_slide:
                    self.parent_slide[parent_slide] = [idx]
                    self.parent_slide_name.append(parent_slide)
                else:
                    self.parent_slide[parent_slide].append(idx)
            
            if self.get_tile_position is not None:
                self.tile_position.append(self.get_tile_position(pth))
            self.idx_in_class[cls] = np.append(self.idx_in_class[cls], idx)
        self.tile_position = np.array(self.tile_position) if self.tile_position else self.tile_position