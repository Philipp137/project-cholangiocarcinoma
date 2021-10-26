from ml4medical.dataset import TileImageDataset
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt

def make_datasets(root_folder = '/work/nb671233/data/CCC_01/CCC/', cohorts=['aachen', 'mainz'], train_val='val'):
    
    return {
        cohort: TileImageDataset(
            root_folder=root_folder + cohort,
            mode=train_val,
            data_variant='CCC',
            normalize=True,
            return_slide_number=False,
        )
        for cohort in cohorts
    }


def show_samples(ds, cohorts, cls=None, all_transforms = False, default_transforms=True, n_slides=3, n_samples=25):
    all_cohort_samples = []
    for cohort in cohorts:
        val_dataset = ds[cohort]
        if cls is None:
            slides = np.random.choice(list(val_dataset.parent_slide.keys()), n_slides)
            cls = [val_dataset.parent_slide_class[np.argmax(slide == np.array(val_dataset.parent_slide_name))] for slide in slides]
        else:
            cls = val_dataset.class_to_idx[cls] if isinstance(cls, str) else cls
            slides = np.random.choice(list(val_dataset.parent_slide_per_class[cls].keys()), n_slides)
            cls = [cls] * n_slides
        val_dataset.apply_transforms = all_transforms
        val_dataset.apply_default_transforms = default_transforms
        sample_idxs = [np.random.choice(list(val_dataset.parent_slide[slide]), n_samples) for slide in slides]

        sample_idxs = np.concatenate(sample_idxs)
        samples = torch.split(val_dataset[sample_idxs][0], n_samples, 0)
        val_dataset.apply_transforms = True
        val_dataset.apply_default_transforms = False
        slide_grids = [make_grid(slide_samples, int(np.ceil(np.sqrt(n_samples)))) for slide_samples in samples]
        
        all_cohort_samples.append(make_grid(torch.stack(slide_grids), n_slides))

    all_samples = make_grid(torch.stack(all_cohort_samples), 1)
    plt.figure()
    plt.imshow(all_samples.permute([1, 2, 0]))
    
    
def main():
    data_folder = '/work/nb671233/data/CCC_01/CCC/'
    # data_folder = 'D:/Arbeit/med_data/CCC/'
    data_cohorts = ['mainz', 'aachen']
    
    tv = 'train'
    
    datasets = make_datasets(data_folder, data_cohorts, tv)
    show_samples(datasets, ['mainz', 'mainz', 'mainz'], 0)
    show_samples(datasets, ['mainz', 'mainz', 'mainz'], 1)
    show_samples(datasets, ['aachen', 'aachen', 'aachen'], 0)
    show_samples(datasets, ['aachen', 'aachen', 'aachen'], 1)
    
    
if __name__ == '__main__':
    main()
