import csv
import os
import shutil
import numpy as np
from pathlib import Path



def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)

root_path = get_project_root()
data_path = root_path + 'CCC/'
csv_path = root_path + 'CCC-labels.csv'


def sort_into_classes_by_csv(data_path, csv_path):
    for cls in ['pos', 'neg', 'missing']:
        cls_path = data_path + cls + '/'
        if not os.path.isdir(cls_path):
            os.mkdir(cls_path)
    
    # sort into classes:
    slide_folders = os.listdir(data_path)
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            slide = row['scan']
            cls = row['label']
            if slide in slide_folders:
                shutil.move(data_path + slide, data_path + cls + '/' + slide)
            #else:
            #    shutil.move(data_path + slide, data_path + 'missing/' + slide)


def train_val_split_slides(data_path, classes, split_fraction=0.125, rand_seed=0, balanced_val_set=True, data_variant='CCC'):
    """
    pick random slides, so that the split_fraction is approximately met for the number of tiles and sort them into train and val folder
    """
    np.random.seed(rand_seed)
    data_path = data_path + '/' if not data_path.endswith('/') else data_path
    n_tiles_in_slide = dict()
    n_tiles_in_class = dict()
    n_slides = dict()
    slides_in_cls = dict()
    for cls in classes:
        n_tiles_in_slide[cls] = []
        if data_variant == 'CCC':
            slides_in_cls[cls] = os.listdir(data_path + cls + '/')
            n_slides[cls] = 0
            for slide in slides_in_cls[cls]:
                n_tiles_in_slide[cls].append(len(os.listdir(data_path + cls + '/' + slide + '/tiles')))
                n_slides[cls] += 1
            n_tiles_in_slide[cls] = np.array(n_tiles_in_slide[cls])
            n_tiles_in_class[cls] = np.sum(n_tiles_in_slide[cls])
        elif data_variant == 'MSIMSS':
            all_tiles_in_slide = os.listdir(data_path + cls + '/')
            slides ={}
            for tile in all_tiles_in_slide:
                slide = '-'.join(tile.split('-')[2:5])
                if slide not in slides:
                    slides[slide] = []
                slides[slide].append(tile)
            slides_in_cls[cls] = slides
            n_slides[cls] = 0
            for slide in slides_in_cls[cls]:
                n_tiles_in_slide[cls].append(len(slides_in_cls[cls][slide]))
                n_slides[cls] += 1
            n_tiles_in_slide[cls] = np.array(n_tiles_in_slide[cls])
            n_tiles_in_class[cls] = np.sum(n_tiles_in_slide[cls])
            
                
            
    
    n_tiles_in_class_val = {cls: round(split_fraction * n_tiles) for cls, n_tiles in n_tiles_in_class.items()}
    if balanced_val_set:
        n_tiles_in_class_val = {cls: min(n_tiles_in_class_val.values()) for cls in classes}
    
    # pick random slides:
    something_wrong = False
    val_slides = dict()
    val_n_tiles = dict()
    for cls in classes:
        n_tiles_so_far = 0
        keep_searching = True
        random_slides = []
        slides_to_pick_from = np.arange(n_slides[cls])
        it = 0
        while keep_searching:
            random_slide = np.random.choice(slides_to_pick_from)
            n_tiles_in_random_slide = n_tiles_in_slide[cls][random_slide]
            random_slides.append(random_slide)
            n_tiles_so_far += n_tiles_in_random_slide
            slides_to_pick_from = list(set(slides_to_pick_from) - set(random_slides))
            if np.min(n_tiles_in_slide[cls]) <= n_tiles_in_class_val[cls] - n_tiles_so_far < np.max(n_tiles_in_slide[cls]):
                random_slide = np.abs((n_tiles_in_class_val[cls] - n_tiles_so_far) - n_tiles_in_slide[cls][slides_to_pick_from]).argmin()
                random_slides.append(slides_to_pick_from[random_slide])
                n_tiles_in_random_slide = n_tiles_in_slide[cls][slides_to_pick_from[random_slide]]
                n_tiles_so_far += n_tiles_in_random_slide
                keep_searching = False
            it += 1
            if it > 100:
                keep_searching = False
                something_wrong = True
        
        val_slides[cls] = random_slides
        val_n_tiles[cls] = n_tiles_so_far
    if not something_wrong:
        # move slides:
        for folder in ['train', 'val']:
            if not os.path.isdir(data_path + folder):
                os.mkdir(data_path + folder)
        
        for cls in classes:
            for slide_idx in val_slides[cls]:
                if not os.path.isdir(data_path + 'val/' + cls + '/'):
                    os.mkdir(data_path + 'val/' + cls + '/')
                if data_variant == 'CCC':
                    shutil.move(data_path + cls + '/' + slides_in_cls[cls][slide_idx],
                                data_path + 'val/' + cls + '/' + slides_in_cls[cls][slide_idx])
                elif data_variant == 'MSIMSS':
                    slide = list(slides_in_cls[cls].keys())[slide_idx]
                    for tile in slides_in_cls[cls][slide]:
                        shutil.move(data_path + cls + '/' + tile, data_path + 'val/' + cls + '/' + tile)
            shutil.move(data_path + cls, data_path + 'train/' + cls)


def copy_code_base(root_path, dest_path, config_file_name):
    shutil.copytree(root_path + '\\ml4medical', dest_path + '\\code\\ml4medical')
    shutil.copy2(root_path + '\\main.py', dest_path + '\\code\\main.py')
    shutil.copy2(root_path + '\\' + config_file_name, dest_path + '\\code\\' + config_file_name)
    
    
def get_checkpoint_path(checkpoint):
    if checkpoint:
        if not checkpoint.endswith('.ckpt'):
            if not checkpoint.endswith('/') and not checkpoint.endswith('\\'):
                checkpoint += '/'
            checkpoint = checkpoint + 'checkpoints/'
            ckpt_files = [f for f in os.listdir(checkpoint) if f.endswith('.ckpt')][0]
            checkpoint += ckpt_files
    return checkpoint


def get_layers_list(module):
    layers_list = []
    for layer in module.children():
        layer_children = [c for c in layer.children()]
        if layer_children:
            layers_list.extend(get_layers_list(layer))
        else:
            layers_list.append(layer)
    return layers_list
    
