from torchvision import models
import torch.nn as nn


class ResNet(nn.Sequential):
    def __init__(self, variant='resnet18', num_classes=2, activate_logits=False, pretrained=False):
        """
        Build one of the resnet variants in torchvision.models.resnet
        """
        super(ResNet, self).__init__()
        self.add_module('nn', getattr(models, variant)(num_classes=num_classes, pretrained=pretrained))
        if activate_logits:
            self.add_module('relu', nn.ReLU())


# maybe more to come
