import torch
from torch.nn import functional as F
import torchmetrics
from torchmetrics.functional.classification import auroc, accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

        
class Classifier(pl.LightningModule):
    def __init__(self, classifier_net, num_classes=2, relevance_class=False, optimizer={'AdamW': {'lr': 1e-5}},
                 patient_level_vali=True, learn_dec_bound=False, batch_size=0, subbatch_size=0, val_batch_size=0, val_subbatch_size=0,
                 subbatch_mean='logits'):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.classifier_net = classifier_net
        self.tile_decision_boundary = torch.nn.Parameter(torch.tensor(0.5), requires_grad=( False and
            (isinstance(learn_dec_bound, bool) and learn_dec_bound) or (isinstance(learn_dec_bound, list) and 'tile' in learn_dec_bound)))
        self.slide_decision_boundary = torch.nn.Parameter(torch.tensor(0.5), requires_grad=(
            (isinstance(learn_dec_bound, bool) and learn_dec_bound) or (isinstance(learn_dec_bound, list) and 'slide' in learn_dec_bound)))
        self.num_classes = num_classes
        self.relevance_class = relevance_class
        self.optimizer_settings = optimizer
        # batch_sizes are only put here so they are logged in tensorboard as hparams
        self.batch_size = batch_size
        self.subbatch_size = subbatch_size
        self.val_batch_size = val_batch_size
        self.val_subbatch_size = val_subbatch_size
        self.subbatch_mean = subbatch_mean if subbatch_size else None
        self.prob_activation = torch.nn.Sigmoid() if num_classes == 1 else torch.nn.Softmax(dim=-1)
        if not self.subbatch_mean or self.subbatch_mean == 'logits':
            self.classification_loss = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()
        elif self.subbatch_mean == 'probs':
            self.classification_loss = torch.nn.BCELoss() if num_classes == 1 else torch.nn.NLLLoss()
        self.accuracy = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.confusion = torchmetrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False,
                                                      dist_sync_on_step=True)
        if self.relevance_class:
            self.confusion_hard = torchmetrics.ConfusionMatrix(num_classes=max(num_classes, 2), normalize='true', compute_on_step=False,
                                                               dist_sync_on_step=True)
        self.patient_level_vali = patient_level_vali
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
            logits = logits.view([*x_shape[:2], logits.shape[-1]])  # unstack subbatch dimension.
            
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
            has_subbatch = True
            # all labels in subbatch have to be the same for num_classes >  1! mean of different labels does not work, since cross_entropy
            # expects dtype long
            y = y[:, 0]
        else:
            has_subbatch = False
            if self.num_classes == 1:
                y = y.float()
                
        pred = self.logits(x)
        if has_subbatch:
            pred = self.get_accumulated_prediction(pred, method='mean_' + self.subbatch_mean, accum_dim=1,
                                                   make_prob=self.subbatch_mean == 'probs')
        # logits = self.accumulated_logits(x, ignore_irrelevant='soft')
        pred = pred.squeeze(-1) if self.num_classes == 1 else pred
        loss = self.classification_loss(pred, y) if not self.subbatch_mean == 'probs' else self.classification_loss(torch.log(pred), y)
        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True, )
        self.log('train/acc', self.accuracy(pred, y.long()),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        has_subbatch = False
        if y.ndim > (1 + self.patient_level_vali): # batch has subbatch dimension
            has_subbatch = True
            y = y[:, 0, ...]
            
        if self.patient_level_vali:
            target = y[..., 0]
        else:
            target = y
    
        if self.num_classes == 1:
            target = target.float()

        pred = self.logits(x)
        if has_subbatch:
            pred = self.get_accumulated_prediction(pred, method='mean_' + self.subbatch_mean, accum_dim=1,
                                                   make_prob=self.subbatch_mean == 'probs')
        # logits = self.accumulated_logits(x, ignore_irrelevant='soft')
        pred = pred.squeeze(-1) if self.num_classes == 1 else pred
        
        if not self.trainer.sanity_checking:
            self.confusion(pred, target.long())
            
            results = torch.cat([pred, y], dim=1)
            self.all_val_results.append(results)
            
    
            # if self.relevance_class:
                # logits_hard = self.accumulated_logits(x, ignore_irrelevant='hard')
                # logits_hard = logits_hard.squeeze(-1) if self.num_classes == 1 else logits_hard
                # loss_hard = self.classification_loss(logits_hard, target)
                #
                # self.log('valid_loss_hard', loss_hard, prog_bar=False, logger=False, sync_dist=True)
                # acc_hard = self.accuracy(self.prob_activation(logits_hard), target.long())
                # self.log('valid_acc_hard', acc_hard, prog_bar=False, logger=False, sync_dist=True)
                # auc_hard = self.auc(self.prob_activation(logits_soft), target.long())
                # self.log('valid_auc_hard', auc_hard, prog_bar=True, sync_dist=True)
                # self.confusion_hard(self.prob_activation(logits_hard), target.long())
                
                
    def get_accumulated_prediction(self, preds, method, accum_dim=0, pos_decision_boundary=0.5, make_prob=True):
        if method == 'mean_labels':
            if make_prob:
                preds = self.prob_activation(preds)
            pos_probs = preds if self.num_classes == 1 else preds[..., 1]
            predicted_labels = (pos_probs >= pos_decision_boundary).float()
            accum_pos_prob = predicted_labels.mean(dim=accum_dim)
            accum_pred = torch.stack([1 - accum_pos_prob, accum_pos_prob], dim=-1) if self.num_classes == 2 else accum_pos_prob
        elif method == 'mean_probs':
            if make_prob:
                preds = self.prob_activation(preds)
            accum_pred = preds.mean(dim=accum_dim)
        elif method == 'mean_logits':
            accum_pred = preds.mean(dim=accum_dim)
            if make_prob:
                accum_pred = self.prob_activation(accum_pred)
        return accum_pred
    
    def validation_epoch_end(self, outputs):
        if self.trainer.sanity_checking:
            return
        all_val_results = torch.cat(self.all_val_results, dim=0)
        all_preds = all_val_results[..., :-2]
        all_probs = all_preds if self.val_subbatch_size and self.subbatch_mean == 'probs' else self.prob_activation(all_preds)
        all_targets = all_val_results[..., -2].long()
        all_slide_ns = all_val_results[..., -1].long()
        methods = ['mean_labels', 'mean_probs', 'mean_logits']
        all_slides_probs_by_method = [[] for _ in range(len(methods))]
        all_slides_targets = []
        if self.patient_level_vali:
            for n in range(all_slide_ns.max()):
                slide_idxs = all_slide_ns == n
                if sum(slide_idxs) > 0 and (self.val_subbatch_size > 10 or sum(slide_idxs) >= 10):
                    for m, method in enumerate(methods):
                        all_slides_probs_by_method[m].append(self.get_accumulated_prediction(all_preds[slide_idxs], method=method,
                            make_prob=self.subbatch_mean != 'probs' or not self.val_subbatch_size))
                    all_slides_targets.append(all_targets[slide_idxs][0])
            all_slides_probs_by_method = torch.stack([torch.stack(preds, 0) for preds in all_slides_probs_by_method], dim=0)
            all_slides_targets = torch.stack(all_slides_targets, dim=0)
            
            for m, method in enumerate(methods):
                all_slides_probs = all_slides_probs_by_method[m, ...]
                pl_tps = all_slides_probs[all_slides_targets == 1, 1].median()
                pl_fps = all_slides_probs[all_slides_targets == 0, 1].median()
                pl_tpr = all_slides_probs[all_slides_targets == 1, 1].round().mean()
                pl_fpr = all_slides_probs[all_slides_targets == 0, 1].round().mean()
                
                pl_auroc = auroc(all_slides_probs, all_slides_targets, num_classes=self.num_classes)
                pl_acc = accuracy(all_slides_probs, all_slides_targets)
                self.all_val_results = []
                self.log('pl_valid_' + method + '/auroc', pl_auroc, prog_bar=False, sync_dist=True)
                self.log('pl_valid_' + method + '/acc', pl_acc, prog_bar=False, sync_dist=True)
                self.log('pl_valid_' + method + '/pos_score', pl_tps, prog_bar=False, sync_dist=True)
                self.log('pl_valid_' + method + '/f_pos_score', pl_fps, prog_bar=False, sync_dist=True)
                self.log('pl_valid_' + method + '/tpr', pl_tpr, prog_bar=False, sync_dist=True)
                self.log('pl_valid_' + method + '/fpr', pl_fpr, prog_bar=False, sync_dist=True)

        tps = all_probs[all_targets == 1, 1].median()
        fps = all_probs[all_targets == 0, 1].median()
        tpr = all_probs[all_targets == 1, 1].round().mean()
        fpr = all_probs[all_targets == 0, 1].round().mean()
        
        loss = self.classification_loss(torch.log(all_probs), all_targets) if self.subbatch_mean == 'probs' else \
               self.classification_loss(all_preds, all_targets)
        confmat = self.confusion.compute()
        auc = auroc(all_preds, all_targets, num_classes=self.num_classes)
        acc = accuracy(all_preds, all_targets)

        self.log('valid/loss', loss, prog_bar=True, sync_dist=True)
        self.log('valid/pos_score', tps, prog_bar=True, sync_dist=True)
        self.log('valid/f_pos_score', fps, prog_bar=True, sync_dist=True)
        self.log('valid/tpr', tpr, prog_bar=True, sync_dist=True)
        self.log('valid/fpr', fpr, prog_bar=True, sync_dist=True)
        self.log('valid/acc', acc, prog_bar=True, sync_dist=True)
        self.log('valid/auc', auc, prog_bar=True, sync_dist=True)
        
        name_extension = '_soft' if self.relevance_class else ''
        self.logger.experiment.add_text('confusion'+name_extension,
                                        f'0-0: {confmat[0, 0].item():.3f};    0-1: {confmat[0, 1].item():.3f};    '
                                        f'1-1: {confmat[1, 1].item():.3f};    1-0: {confmat[1, 0].item():.3f}',
                                        self.current_epoch)
        self.confusion.reset()
        
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