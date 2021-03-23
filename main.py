from net import ResNet
from model import Classifier
from dataset import TileImageDataset, TileSubBatchSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

random_seed = 42
data_folder = 'D:/Arbeit/cholangiocarcinoma-data/CRC_KR/'
num_classes = 1
relevance_class = False
train_batch_size = 384
train_subbatch_size = 0
train_num_workers = 4
val_batch_size = 12
val_subbatch_size = 96
val_num_workers = 4
num_nodes = 1


if __name__ =="__main__":
    
    if torch.cuda.device_count() > 1 or num_nodes > 1:
        distributed = True
    
    pl.seed_everything(random_seed)
    train_dataset = TileImageDataset(root_folder=data_folder, mode='train', data_variant='MSIMSS', normalize=False)
    train_sampler = TileSubBatchSampler(train_dataset, subbatch_size=train_subbatch_size, mode='class', shuffle=True, balance=None,
                                        distributed=distributed)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, num_workers=train_num_workers,
                              persistent_workers=True)
    
    val_dataset = TileImageDataset(root_folder=data_folder, mode='val', data_variant='MSIMSS', normalize=False)
    val_sampler = TileSubBatchSampler(val_dataset, subbatch_size=val_subbatch_size, mode='slide', shuffle=False,
                                      distributed=distributed)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, num_workers=val_num_workers,
                            persistent_workers=True)
    
    model = Classifier(ResNet(variant='resnet18', num_classes=num_classes + relevance_class),
                       num_classes=num_classes,
                       relevance_class=relevance_class,
                       lr=1e-2)
    
    trainer = pl.Trainer(gpus=-1, num_nodes=num_nodes, benchmark=True, max_epochs=50, replace_sampler_ddp=False, accelerator='ddp')
    trainer.fit(model, train_loader, val_loader, )