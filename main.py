from ml4medical import net
from ml4medical.model import Classifier
from ml4medical.dataset import DataModule
from ml4medical.utils import copy_code_base, get_checkpoint_path
import pytorch_lightning as pl
import json
import torch
import argparse
import os
from ml4medical.utils import get_project_root


if __name__ =="__main__":
    config_file_name = 'config_CCC.json'
    from_shell = True
    args = None
    this_dir = get_project_root()
    print("Hello this is ml4medical")
    #this_dir = "/home/nb671233/project-cholangiocarcinoma"
    if from_shell:
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', '--num_workers', type=int, help='Number of workers for the DataLoader to use', default=None)
        parser.add_argument('-n', '--num_nodes', type=int, help='Number of nodes to use on the cluster', default=None)
        parser.add_argument('-c', '--config', type=str, help='Name of config file to use (including path if in a different folder)', default=None)
        parser.add_argument('-r', '--resume', type=str, help='Path to version folder to resume training from or to checkpoint file ('
                                                             'including path if in a different folder)', default=None)

        parser.add_argument('-rc', '--resume_config', type=bool, help='Whether or not to use the config from the resume folder',
                            default=False)
        parser.add_argument('-d', '--data', type=str, help='name of data type ( either  "CCC" or "MSIMSS" )' , default=None)
        args = parser.parse_args()
        
        #if args.resume_config:
        
        config_name = args.config or this_dir + '/config.json'
    else:
        config_name = this_dir + '/' + config_file_name

    with open(config_name) as file:
        config = json.load(file)
    print("config file: ", config_name) 
    trainer_conf = config['trainer']
    num_workers = args.num_workers or trainer_conf['num_workers'] if args is not None else trainer_conf['num_workers']
    num_nodes = args.num_nodes or trainer_conf['num_nodes'] if args is not None else trainer_conf['num_nodes']
    data_variant = args.data or trainer_conf['data_variant'] if args is not None else trainer_conf['data_variant']
    if 'resume' in trainer_conf:
        resume = args.resume or trainer_conf['resume'] if args is not None else trainer_conf['resume']
    else:
        resume = args.resume if args is not None else None
    
    data_folder = config['data_root_paths'][trainer_conf['cluster']] + config['data_folder'][trainer_conf['data_variant']]
    
    n_gpu = torch.cuda.device_count()
    print(f'seeing {n_gpu} GPU')
    distributed = n_gpu > 1 or num_nodes > 1
    
    pl.seed_everything(trainer_conf['random_seed'])
    data_module = DataModule(
            root_folder=data_folder,
            data_variant=data_variant,
            augmentations = trainer_conf['augmentations'],
            train_batch_size=trainer_conf['batch_size'],
            val_batch_size=trainer_conf['val_batch_size'],
            train_subbatch_size=trainer_conf['subbatch_size'],
            val_subbatch_size=trainer_conf['val_subbatch_size'],
            val_return_slide_number=True,
            train_balance=trainer_conf['balance'],
            val_balance=trainer_conf['balance'],
            distributed=distributed,
            num_workers=num_workers,
            persistent_workers=True
    )
    
    model_conf = config['model']
    classifier_net = getattr(net, model_conf['net_type'])(variant=model_conf['variant'],
                                                          num_classes=model_conf['num_classes'] + model_conf['relevance_class'],
                                                          pretrained=model_conf['pretrained'],
                                                          dropout=model_conf['dropout']
                                                         )
                   
    if 'freeze_until' in config['model'] and config['model']['freeze_until']:
        classifier_net.freeze_until_nth_layer(config['model']['freeze_until'])
        
    model = Classifier(
            classifier_net,
            num_classes=model_conf['num_classes'],
            relevance_class=model_conf['relevance_class'],
            optimizer=model_conf['optimizer'],
            patient_level_vali=True,
            batch_size=trainer_conf['batch_size'],
            val_batch_size=trainer_conf['val_batch_size'],
            subbatch_size=trainer_conf['subbatch_size'],
            val_subbatch_size=trainer_conf['val_subbatch_size'],
            subbatch_mean=model_conf['subbatch_mean'],
            augmentations=trainer_conf['augmentations'],
    )
    
    accelerator = 'ddp' if distributed else None
    
    checkpoint = get_checkpoint_path(resume) or None

    trainer = pl.Trainer(gpus=-1,
                         num_nodes=num_nodes,
                         max_epochs=trainer_conf['epoches'],
                         precision=trainer_conf['precision'],
                         benchmark=True,
                         replace_sampler_ddp=False,
                         accelerator=accelerator,
                         default_root_dir=this_dir,
                         fast_dev_run=False,
                         resume_from_checkpoint=checkpoint
                         )
    #if int(os.environ["LOCAL_RANK"]) == 0:
    #    copy_code_base(this_dir, trainer.logger.log_dir, config_file_name)


    trainer.fit(model, data_module)
