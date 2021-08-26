import torch
from torch.nn import functional as F
import torchmetrics
from torchmetrics.functional.classification import auroc, accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

        
class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, num_classes=2, relevance_class=False, optimizer={'AdamW': {'lr': 1e-5}}, lr=1e-3,
                 patient_level_vali=True):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net
        self.num_classes = num_classes
        self.relevance_class = relevance_class
        self.optimizer_settings = optimizer
        self.lr = lr
        self.prob_activation = torch.nn.Sigmoid() if num_classes == 1 else torch.nn.Softmax(dim=-1)
        self.classification_loss = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.auc = torchmetrics.AUROC(num_classes=2, compute_on_step=False, dist_sync_on_step=True, pos_label=1)
        self.confusion = torchmetrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False,
                                                      dist_sync_on_step=True)
        if self.relevance_class:
            self.confusion_hard = torchmetrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False,
                                                               dist_sync_on_step=True)
        self.patient_level_vali = patient_level_vali
        if patient_level_vali:
            self.all_val_results = []

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
        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True, )
        self.log('train/acc', self.accuracy(logits, y.long()),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.patient_level_vali and not self.trainer.sanity_checking:
            target = y[..., 0]
        else:
            target = y
        # if len(y.shape) > 1: # batch has subbatch dimension
        #     y = y.float().mean(1)
        #     if self.num_classes > 1:
        #         y = y.long()# all labels in subbatch habe to be the same! mean of different labels does not work, since cross_entropy
        #                     # expects dtype long
        if target.ndim > 1: # batch has subbatch dimension
            # all labels in subbatch have to be the same for num_classes >  1! mean of different labels does not work, since cross_entropy
            # expects dtype long
            target = target.float().mean(1) if self.num_classes == 1 else target[:, 0]
        elif self.num_classes == 1:
            target = target.float()
                
        logits_soft = self.accumulated_logits(x, ignore_irrelevant='soft')
        logits_soft = logits_soft.squeeze(-1) if self.num_classes == 1 else logits_soft
        loss_soft = self.classification_loss(logits_soft, target)
        
        if not self.trainer.sanity_checking:
            acc = self.accuracy(logits_soft, target.long())
            self.auc(logits_soft, target.long())
            self.confusion(logits_soft, target.long())
            
            if self.patient_level_vali:
                results = torch.cat([logits_soft, y], dim=1)
                self.all_val_results.append(results)
            name_extension = '_soft' if self.relevance_class else ''
            self.log('valid/loss'+name_extension, loss_soft, prog_bar=True, sync_dist=True)
            self.log('valid/acc'+name_extension, acc, prog_bar=True, sync_dist=True)
            
    
            if self.relevance_class:
                logits_hard = self.accumulated_logits(x, ignore_irrelevant='hard')
                logits_hard = logits_hard.squeeze(-1) if self.num_classes == 1 else logits_hard
                loss_hard = self.classification_loss(logits_hard, target)
        
                #self.log('valid_loss_hard', loss_hard, on_step=True, prog_bar=False)
                self.log('valid_loss_hard', loss_hard, prog_bar=False, logger=False, sync_dist=True)
                acc_hard = self.accuracy(self.prob_activation(logits_hard), target.long())
                #self.log('valid_acc_hard', acc_hard, on_step=True, prog_bar=False)
                self.log('valid_acc_hard', acc_hard, prog_bar=False, logger=False, sync_dist=True)
                auc_hard = self.auc(self.prob_activation(logits_soft), target.long())
                self.log('valid_auc_hard', auc_hard, prog_bar=True, sync_dist=True)
                self.confusion_hard(self.prob_activation(logits_hard), target.long())
                
    def get_accumulated_prediction(self, logits, method, accum_dim=0, pos_decision_boundary=0.5):
        if method == 'mean_labels':
            probs = self.prob_activation(logits)
            pos_probs = probs if self.num_classes == 1 else probs[..., 1]
            predicted_labels = (pos_probs >= pos_decision_boundary).float()
            accum_pos_prob = predicted_labels.mean(dim=accum_dim)
            accum_pred = torch.stack([1 - accum_pos_prob, accum_pos_prob], dim=-1)
        elif method == 'mean_probs':
            probs = self.prob_activation(logits)
            accum_pred = probs.mean(dim=accum_dim)
        elif method == 'mean_logits':
            accum_pred = logits.mean(dim=accum_dim)
        return accum_pred
    
    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking:
            return
        all_slides_preds = []
        all_slides_targets = []
        if self.patient_level_vali:
            all_val_results = torch.cat(self.all_val_results, dim=0)
            all_logits = all_val_results[..., :-2]
            all_targets = all_val_results[..., -2].long()
            all_slide_ns = all_val_results[..., -1].long()
            for n in range(all_slide_ns.max()):
                slide_idxs = all_slide_ns == n
                if sum(slide_idxs) >= 10:
                    all_slides_preds.append(self.get_accumulated_prediction(all_logits[slide_idxs], method='mean_labels'))
                    all_slides_targets.append(all_targets[slide_idxs][0])
            all_slides_preds = torch.tensor(all_slides_preds)
            all_slides_targets = torch.tensor(all_slides_targets)
            tpr = self.prob_activation(all_logits[all_targets == 1]).round().mean()
            fpr = self.prob_activation(all_logits[all_targets == 0]).round().mean()
            pl_tpr = all_slides_preds[all_slides_targets == 1].round().mean()
            pl_fpr = all_slides_preds[all_slides_targets == 0].round().mean()
            
            pl_auroc = auroc(all_slides_preds, all_slides_targets, pos_label=1)
            pl_acc = accuracy(all_slides_preds, all_slides_targets)
            self.all_val_results = []
            self.log('pl_valid/auc', pl_auroc, prog_bar=True, sync_dist=True)
            self.log('pl_valid/acc', pl_acc, prog_bar=True, sync_dist=True)
            self.log('pl_valid/tpr', pl_tpr, prog_bar=True, sync_dist=True)
            self.log('pl_valid/fpr', pl_fpr, prog_bar=True, sync_dist=True)
            self.log('valid/tpr', tpr, prog_bar=True, sync_dist=True)
            self.log('valid/fpr', fpr, prog_bar=True, sync_dist=True)
            
        confmat = self.confusion.compute()
        auc = self.auc.compute()
        
        self.log('valid/auc', auc, prog_bar=True, sync_dist=True)
        
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
            
    # for now, just do the same in test as in validation
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        optimizer_name = list(self.optimizer_settings.keys())[0]
        optimizer_settings = self.optimizer_settings[optimizer_name]
        if not isinstance(optimizer_settings['lr'], list):
            lr_classifier = optimizer_settings['lr']
        else:
            lr_classifier = optimizer_settings['lr'][1]
            optimizer_settings['lr'] = optimizer_settings['lr'][0]
        optimizer = getattr(torch.optim, optimizer_name)(
                [{'params': self.classifier_net.params_base},
                 {'params': self.classifier_net.params_classifier, 'lr': lr_classifier}
                 ], **optimizer_settings)
        
        # scheduler = ReduceLROnPlateau(optimizer, patience=3)
        # name_extension = '_soft' if self.relevance_class else ''
        # return dict(optimizer=optimizer, lr_scheduler=scheduler, monitor='valid_loss'+name_extension)
        
        return optimizer