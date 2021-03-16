import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, num_classes=2, relevance_class=False, lr=1e-3):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net
        self.num_classes = num_classes
        self.relevance_class = relevance_class
        self.lr = lr
        self.prob_activation = torch.nn.Sigmoid() if num_classes == 1 else torch.nn.Softmax(dim=-1)
        self.classification_loss = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.auc = pl.metrics.AUROC(num_classes=max(num_classes, 2), average='weighted', pos_label=0, compute_on_step=False)
        self.confusion = pl.metrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False)
        if self.relevance_class:
            self.confusion_hard = pl.metrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False)

    def forward(self, x):
        # use forward for inference/predictions
        probs = self.logits(x)
        if self.relevance_class:
            probs[..., -1] = torch.sigmoid(probs[..., -1])
            probs[..., :-1] = self.prob_activation(probs[..., :-1])
        else:
            probs = self.prob_activation(probs)
            
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
        if not self.relevance_class and logits.ndim > 2:
            logits = logits.mean(-2)  # mean over subbatch dimension if present, else mean over batch dimension (for inference).
        elif self.relevance_class:
            # relevance is modeled as independent class with sigmoid. In this case, the relevance rating goes "ontop" of the classification
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
        
        if y.ndim > 1: # batch has subbatch dimension
            # all labels in subbatch have to be the same for num_classes >  1! mean of different labels does not work, since cross_entropy
            # expects dtype long
            y = y.float().mean(1) if self.num_classes == 1 else y[:, 0]
        else:
            if self.num_classes == 1:
                y = y.float()
        
        logits = self.accumulated_logits(x, ignore_irrelevant='soft')
        logits = logits.squeeze(-1) if self.num_classes == 1 else logits
        loss = self.classification_loss(logits, y)
        
        self.log('train_loss', loss, on_epoch=True, )
        self.log('train_acc', self.accuracy(self.prob_activation(logits), y.long()), on_epoch=True, prog_bar=False)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if len(y.shape) > 1: # batch has subbatch dimension
            y = y.float().mean(1)
            if self.num_classes > 1:
                y = y.long()# all labels in subbatch habe to be the same! mean of different labels does not work, since cross_entropy
                            # expects dtype long
        
        logits_soft = self.accumulated_logits(x, ignore_irrelevant='soft')
        logits_soft = logits_soft.squeeze(-1) if self.num_classes == 1 else logits_soft
        loss_soft = self.classification_loss(logits_soft, y)
        
        if not self.trainer.running_sanity_check:
            name_extension = '_soft' if self.relevance_class else ''
            #self.log('valid_loss'+name_extension, loss_soft, on_step=True, prog_bar=False)
            self.log('valid_loss'+name_extension, loss_soft, prog_bar=True)
            acc = self.accuracy(self.prob_activation(logits_soft), y.long())
            #self.log('valid_acc'+name_extension, acc_soft, on_step=True, prog_bar=False)
            self.log('valid_acc'+name_extension, acc, prog_bar=True)
            self.auc(self.prob_activation(logits_soft), y.long())
            
            self.confusion(self.prob_activation(logits_soft), y.long())
    
            if self.relevance_class:
                logits_hard = self.accumulated_logits(x, ignore_irrelevant='hard')
                logits_hard = logits_hard.squeeze(-1) if self.num_classes == 1 else logits_hard
                loss_hard = self.classification_loss(logits_hard, y)
        
                #self.log('valid_loss_hard', loss_hard, on_step=True, prog_bar=False)
                self.log('valid_loss_hard', loss_hard, prog_bar=False, logger=False)
                acc_hard = self.accuracy(self.prob_activation(logits_hard), y.long())
                #self.log('valid_acc_hard', acc_hard, on_step=True, prog_bar=False)
                self.log('valid_acc_hard', acc_hard, prog_bar=False, logger=False)
                auc_hard = self.auc(self.prob_activation(logits_soft), y.long())
                self.log('valid_auc_hard', auc_hard, prog_bar=True)
                self.confusion_hard(self.prob_activation(logits_hard), y.long())
        
    def validation_epoch_end(self, outputs):
        if not self.trainer.running_sanity_check:
            confmat = self.confusion.compute()
            auc = self.auc.compute()
            self.log('valid_auc', auc, prog_bar=True)
            
            name_extension = '_soft' if self.relevance_class else ''
            self.logger.experiment.add_text('confusion'+name_extension,
                                            f'0-0: {confmat[0, 0].item():.3f};    0-1: {confmat[0, 1].item():.3f};    '
                                            f'1-1: {confmat[1, 1].item():.3f};    1-0: {confmat[1, 0].item():.3f}',
                                            self.current_epoch)
            self.confusion.reset()
            self.auc.reset()
            self.accuracy.reset()
    
            if self.relevance_class:
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        name_extension = '_soft' if self.relevance_class else ''
        return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='valid_loss'+name_extension)