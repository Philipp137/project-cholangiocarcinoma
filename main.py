from net import ResNet
from model import Classifier
from dataset import TileImageDataset, TileSubBatchSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl

random_seed = 42
data_folder = 'D:/Arbeit/cholangiocarcinoma-data/CRC_KR/'
num_classes = 2
relevance_class = False
train_batch_size = 6
train_subbatch_size = 64
num_workers = 12
val_batch_size = 8
val_subbatch_size = 96


if __name__ =="__main__":
    pl.seed_everything(random_seed)
    train_dataset = TileImageDataset(root_folder=data_folder, mode='train', data_variant='MSIMSS', normalize=False)
    train_sampler = TileSubBatchSampler(train_dataset, subbatch_size=train_subbatch_size, mode='class', shuffle=True, balance=None)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, num_workers=num_workers)
    
    val_dataset = TileImageDataset(root_folder=data_folder, mode='val', data_variant='MSIMSS', normalize=False)
    val_sampler = TileSubBatchSampler(val_dataset, subbatch_size=val_subbatch_size, mode='slide', shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, num_workers=num_workers)
    
    model = Classifier(ResNet(variant='resnet18', num_classes=num_classes + relevance_class),
                       num_classes=num_classes,
                       relevance_class=relevance_class,
                       lr=1e-3)
    
    trainer = pl.Trainer(gpus=-1, benchmark=False, max_epochs=5, precision=16)
    trainer.fit(model, train_loader, val_loader)