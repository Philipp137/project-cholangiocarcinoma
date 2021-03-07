from net import ResNet
from model import Classifier
from dataset import TileImageDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

random_seed = 42
data_folder = 'D:/Arbeit/cholangiocarcinoma-data/CRC_KR/'
batch_size = 192
num_workers = 12


if __name__ =="__main__":
    pl.seed_everything(random_seed)
    
    train_loader = DataLoader(TileImageDataset(root_folder=data_folder, mode='train', normalize=False), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(TileImageDataset(root_folder=data_folder, mode='val', normalize=False), batch_size=batch_size,
                            num_workers=num_workers)
    
    model = Classifier(ResNet(variant='resnet18'), lr=1e-3)
    
    trainer = pl.Trainer(gpus=-1, benchmark=True, max_epochs=5, precision=16)
    trainer.fit(model, train_loader, val_loader)