import torch
from torch.nn import functional as F
import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, num_classes=2, relevance_class=False, lr=1e-3):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net
        self.relevance_class = relevance_class
        
        self.accuracy = pl.metrics.Accuracy()
        self.confusion_soft = pl.metrics.ConfusionMatrix(num_classes=num_classes + relevance_class, normalize='true', compute_on_step=False)
        self.confusion_hard = pl.metrics.ConfusionMatrix(num_classes=num_classes + relevance_class, normalize='true', compute_on_step=False)

    def forward(self, x):
        # use forward for inference/predictions
        probs = self.logits(x)
        if self.relevance_class:
            probs[..., -1] = torch.sigmoid(probs[..., -1])
        probs[..., :-1] = F.softmax(probs[..., :-1], dim=-1)
        return probs
    
    def logits(self, x):
        x_shape = x.shape
    
        if len(x_shape) > 4:  # batch has subbatch dimension
            x = x.view([-1, *x.shape[-3:]])  # stack subbatch dimension onto batch dimension
    
        logits = self.classifier_net(x)
    
        if len(x_shape) > 4:  # batch has subbatch dimension
            logits = logits.view([*x_shape[:2], logits.shape[-1]])  # unstack subbatch dimension and mean over it.
            
        return logits
        
    def accumulated_logits(self, x, ignore_irrelevant='soft'):
        logits = self.logits(x)
        if not self.relevance_class:
            logits = logits.mean(-2)  # mean over subbatch dimension if present, else mean over batch dimension (for inference).
        else:
            # model relevance as independent class with sigmoid. In this case, the relevance rating goes "ontop" of the classification
            # output. So the classifier can decide for one of the classes, but rate the prediction as irrelevant.
            # Maybe it could also make sense to model it as an additional dependent class by using softmax. Then the classifier decides
            # for either of the classes or that it is "something else" / irrelevant. For this the implemented "ignoring logic" below
            # might be a bit scetchy and would need rethinking
            relevance = F.sigmoid(logits[..., [-1]])
            if ignore_irrelevant == 'hard':
                relevance = relevance.round()
            # "Ignoring logic": for "soft" ignoring, we "fade in" the logits by the relevance and adjust the mean accordingly,
            # by substracting the sum of (1-relevance) from the number of tiles before deviding. Looking at the "hard" case maybe helps
            # understanding. Here, we first decide wether a tile is fully relevant or irrelevant (by rounding to 0 or 1) and ignore the
            # irrelevant tiles completely when meaning
            logits = (logits[..., :-1] * relevance).sum(-2) / torch.clip(logits.shape[-2] - (1 - relevance).sum(-2), 1)
                
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if len(y.shape) > 1: # batch has subbatch dimension
            y = y[:, 0] # all labels in subbatch habe to be the same! mean of different labels does not work, since cross_entropy expects
            # dtype long
        logits = self.accumulated_logits(x, ignore_irrelevant='soft')
        loss = F.cross_entropy(logits, y)
        
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', self.accuracy(F.softmax(logits, dim=-1), y), on_epoch=True, prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if len(y.shape) > 1: # batch has subbatch dimension
            y = y[:, 0] # all labels in subbatch habe to be the same! mean of different labels does not work, since cross_entropy expects
            # dtype long
        
        logits_soft = self.accumulated_logits(x, ignore_irrelevant='soft')
        loss_soft = F.cross_entropy(logits_soft, y)

        if not self.relevance_class:
            logits_hard = self.accumulated_logits(x, ignore_irrelevant='hard')
            loss_hard = F.cross_entropy(logits_hard, y)

            self.log('valid_loss_soft', loss_soft, on_step=True, prog_bar=False)
            self.log('valid_loss_soft', loss_soft, on_epoch=True, prog_bar=True, logger=False)
            acc_soft = self.accuracy(F.softmax(logits_soft, dim=-1), y)
            self.log('valid_acc_soft', acc_soft, on_step=True, prog_bar=False)
            self.log('valid_acc_soft', acc_soft, on_epoch=True, prog_bar=True, logger=False)
            self.confusion_soft(F.softmax(logits_soft, dim=-1), y)
    
            self.log('valid_loss_hard', loss_hard, on_step=True, prog_bar=False)
            self.log('valid_loss_hard', loss_hard, on_epoch=True, prog_bar=True, logger=False)
            acc_hard = self.accuracy(F.softmax(logits_hard, dim=-1), y)
            self.log('valid_acc_hard', acc_hard, on_step=True, prog_bar=False)
            self.log('valid_acc_hard', acc_hard, on_epoch=True, prog_bar=True, logger=False)
            self.confusion_hard(F.softmax(logits_hard, dim=-1), y)
        
    def validation_epoch_end(self, outputs):
        confmat = self.confusion_soft.compute()
        self.logger.experiment.add_text('confusion_soft',
                                        f'0-0: {confmat[0, 0].item():.3f};    0-1: {confmat[0, 1].item():.3f};    '
                                        f'1-1: {confmat[1, 1].item():.3f};    1-0: {confmat[1, 0].item():.3f}',
                                        self.current_epoch)
        self.confusion_soft.reset()
        
        confmat = self.confusion_hard.compute()
        self.logger.experiment.add_text('confusion_hard',
                                        f'0-0: {confmat[0, 0].item():.3f};    0-1: {confmat[0, 1].item():.3f};    '
                                        f'1-1: {confmat[1, 1].item():.3f};    1-0: {confmat[1, 0].item():.3f}',
                                        self.current_epoch)
        self.confusion_hard.reset()
        
        #TODO show heatmap for some slides
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.classifier_net(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
