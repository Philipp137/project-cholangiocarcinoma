import torch
from torch.nn import functional as F
import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, lr=1e-3):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net
        
        self.accuracy = pl.metrics.Accuracy()
        self.confusion = pl.metrics.ConfusionMatrix(num_classes=2, normalize='true', compute_on_step=False)

    def forward(self, x):
        # use forward for inference/predictions
        probs = F.softmax(self.classifier_net(x), dim=-1)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', self.accuracy(F.softmax(y_hat, dim=-1), y), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True, prog_bar=False)
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True, logger=False)
        acc = self.accuracy(F.softmax(y_hat, dim=-1), y)
        self.log('valid_acc', acc, on_step=True, prog_bar=False)
        self.log('valid_acc', acc, on_epoch=True, prog_bar=True, logger=False)
        self.confusion(F.softmax(y_hat, dim=-1), y)

    def validation_epoch_end(self, outputs):
        confmat = self.confusion.compute()
        self.logger.experiment.add_text('confusion',
                                        f'0-0: {confmat[0, 0].item():.3f};    0-1: {confmat[0, 1].item():.3f};    '
                                        f'1-1: {confmat[1, 1].item():.3f};    1-0: {confmat[1, 0].item():.3f}',
                                        self.current_epoch)
        self.confusion.reset()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
