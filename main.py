from net import ResNet
from model import Classifier
from dataset import DataModule
import pytorch_lightning as pl
import torch

random_seed = 42
data_folder = 'D:/Arbeit/cholangiocarcinoma-data/CRC_KR/'
data_variant = 'MSIMSS'
num_classes = 1
relevance_class = False
train_batch_size = 384
train_subbatch_size = 0
num_workers = 4
val_batch_size = 12
val_subbatch_size = 96
num_nodes = 2


if __name__ =="__main__":
    
    distributed = torch.cuda.device_count() > 1 or num_nodes > 1
    
    pl.seed_everything(random_seed)
    data_module = DataModule(
            root_folder=data_folder,
            data_variant=data_variant,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_subbatch_size=train_subbatch_size,
            val_subbatch_size=val_subbatch_size,
            distributed=distributed,
            num_workers=num_workers,
            persistent_workers=True
    )
    
    model = Classifier(
            ResNet(variant='resnet18', num_classes=num_classes + relevance_class),
            num_classes=num_classes,
            relevance_class=relevance_class,
            lr=1e-3
    )
    accelerator = 'ddp' if distributed else None
    trainer = pl.Trainer(gpus=2, num_nodes=num_nodes, benchmark=True, max_epochs=10, replace_sampler_ddp=False, accelerator=accelerator)
    trainer.fit(model, data_module)