from ml4medical.dataset import TileImageDataset
import matplotlib.pyplot as plt
from ml4medical.visualize import show_heatmap
from ml4medical.utils import get_checkpoint_path
from ml4medical.net import ResNet
from ml4medical.model import Classifier
from ml4medical.utils import get_project_root
import json
import torch


config_file_name = 'config_CCC.json'
args = None
this_dir = get_project_root()

config_name = this_dir + '/' + config_file_name

with open(config_name) as file:
    config = json.load(file)
print("config file: ", config_name)
trainer_conf = config['trainer']
data_variant = args.data or trainer_conf['data_variant'] if args is not None else trainer_conf['data_variant']
if 'resume' in trainer_conf:
    resume = args.resume or trainer_conf['resume'] if args is not None else trainer_conf['resume']
else:
    resume = args.resume if args is not None else None

data_folder = config['data_root_paths'][trainer_conf['cluster']] + config['data_folder'][data_variant]

model_conf = config['model']
classifier_net = ResNet(variant=model_conf['resnet_variant'], num_classes=model_conf['num_classes'] + model_conf['relevance_class'],
                        pretrained=model_conf['pretrained'])

model = Classifier(classifier_net, num_classes=model_conf['num_classes'], relevance_class=model_conf['relevance_class'],
        optimizer=model_conf['optimizer'], patient_level_vali=True, batch_size=trainer_conf['batch_size'],
        val_batch_size=trainer_conf['val_batch_size'], subbatch_size=trainer_conf['subbatch_size'],
        val_subbatch_size=trainer_conf['val_subbatch_size'], subbatch_mean=model_conf['subbatch_mean'],
        augmentations=trainer_conf['augmentations'])

checkpoint = get_checkpoint_path(resume) or None

model = model.load_from_checkpoint(checkpoint)
model.eval()


val_dataset = TileImageDataset(data_folder, 'val', normalize=True, data_variant=data_variant, return_slide_number=False)

for slide_number in range(len(val_dataset.parent_slide_name)):
    all_tiles, all_targetsNslide_ns, all_pos = val_dataset.get_tiles_from_slide(slide_number)
    target = all_targetsNslide_ns[0]

    with torch.no_grad():
        all_preds = model(all_tiles)

    h = show_heatmap(all_pos[:, 0], all_pos[:, 1], all_preds[:, 1])
    plt.title(f'slide: {val_dataset.parent_slide_name[slide_number]}; Target: {target}\n mean_preds: '
              f'{all_preds[:, 1].mean():1.2}; mean_labels: {all_preds[:, 1].round().mean():1.2}')
    plt.colorbar()
    plt.savefig(str(val_dataset.parent_slide_name[slide_number])+"-heat.png")
    plt.close("all")
