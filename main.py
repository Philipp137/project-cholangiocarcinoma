from ml4medical.net import ResNet
from ml4medical.model import Classifier
from ml4medical.dataset import DataModule
import pytorch_lightning as pl
import json
import torch
import argparse
import os


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--num_workers', type=int, help='Number of workers for the DataLoader to use', default=None)
    parser.add_argument('-n', '--num_nodes', type=int, help='Number of nodes to use on te cluster', default=None)
    parser.add_argument('-c', '--config', type=str, help='Name of config file to use (including path if in a different folder)', default=None)
    args = parser.parse_args()
    config_name = args.config or os.path.dirname(os.path.abspath(__file__)) + '/config.json'
    
    with open(config_name) as file:
        config = json.load(file)
        
    
    trainer_conf = config['trainer']
    num_workers = args.num_workers or trainer_conf['num_workers']
    num_nodes = args.num_nodes or trainer_conf['num_nodes']
    data_folder = config['data_root_paths'][trainer_conf['cluster']] + config['data_folder'][trainer_conf['data_variant']]
    
    distributed = torch.cuda.device_count() > 1 or num_nodes > 1
    pl.seed_everything(trainer_conf['random_seed'])
    data_module = DataModule(
            root_folder=data_folder,
            data_variant=trainer_conf['data_variant'],
            train_batch_size=trainer_conf['batch_size'],
            val_batch_size=trainer_conf['val_batch_size'],
            train_subbatch_size=trainer_conf['subbatch_size'],
            val_subbatch_size=trainer_conf['val_subbatch_size'],
            distributed=distributed,
            num_workers=num_workers,
            persistent_workers=True
    )
    
    model_conf = config['model']
    model = Classifier(
            ResNet(variant=model_conf['resnet_variant'], num_classes=model_conf['num_classes'] + model_conf['relevance_class']),
            num_classes=model_conf['num_classes'],
            relevance_class=model_conf['relevance_class'],
            lr=model_conf['learning_rate']
    )
    
    accelerator = 'ddp' if distributed else None
    trainer = pl.Trainer(gpus=-1,
                         num_nodes=num_nodes,
                         max_epochs=trainer_conf['epoches'],
                         precision=trainer_conf['precision'],
                         benchmark=True,
                         replace_sampler_ddp=False,
                         accelerator=accelerator,)
                         
    trainer.fit(model, data_module)