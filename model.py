import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, lr=1e-3):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net

    def forward(self, x):
        # use forward for inference/predictions
        probs = F.softmax(self.classifier_net(x), dim=-1)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
