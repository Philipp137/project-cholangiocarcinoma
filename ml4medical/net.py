from torchvision import models
import torch.nn as nn


class ResNet(nn.Sequential):
    def __init__(self, variant='resnet18', num_classes=2, activate_logits=False, pretrained=False):
        """
        Build one of the resnet variants in torchvision.models.resnet
        """
        super(ResNet, self).__init__()
        if pretrained and (num_classes != 1000):
            self.add_module('nn', getattr(models, variant)(num_classes=1000, pretrained=pretrained))
            self.nn.fc = nn.Linear(self.nn.fc.in_features, num_classes)
        else:
            self.add_module('nn', getattr(models, variant)(num_classes=num_classes, pretrained=pretrained))
        if activate_logits:
            self.add_module('relu', nn.ReLU())


# maybe more to come
