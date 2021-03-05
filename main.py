from net import ResNet
from model import Classifier
from dataset import ImageDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

random_seed = 42
data_folder = 'D:/Arbeit/cholangiocarcinoma-data/'
batch_size = 192


if __name__ =="__main__":
    pl.seed_everything(random_seed)
    
    train_loader = DataLoader(ImageDataset(data_folder=data_folder, mode='train'), batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(ImageDataset(data_folder=data_folder, mode='val'), batch_size=batch_size, num_workers=12)
    
    model = Classifier(ResNet(variant='resnet18'), lr=1e-3)
    
    trainer = pl.Trainer(gpus=1,)
    trainer.fit(model, train_loader, val_loader)