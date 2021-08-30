from torchvision import models
import torch.nn as nn
from ml4medical.utils import get_layers_list


class ResNet(nn.Sequential):
    def __init__(self, variant='resnet18', dropout=0.0, num_classes=2, activate_logits=False, pretrained=False):
        """
        Build one of the resnet variants in torchvision.models.resnet
        """
        self.pretrained = pretrained
        super(ResNet, self).__init__()
        if pretrained and (num_classes != 1000):
            self.add_module('nn', getattr(models, variant)(num_classes=1000, pretrained=pretrained))
            self.nn.fc = nn.Linear(self.nn.fc.in_features, num_classes)
        else:
            self.add_module('nn', getattr(models, variant)(num_classes=num_classes, pretrained=pretrained))
        if activate_logits:
            self.add_module('relu', nn.ReLU())
        
        self.params_base = [p[1] for p in self.nn.named_parameters() if not p[0].startswith('fc')]
        self.params_classifier = [p for p in self.nn.fc.parameters()]
        self.flat_layers_list = get_layers_list(self.nn)
        self.frozen_layers_list = []
        self.learning_layers_list = self.flat_layers_list.copy()
        
    def freeze_until_nth_layer(self, index):
        for layer in self.flat_layers_list[:index]:
            layer.requires_grad_(False)
            self.frozen_layers_list.append(layer)
        self.learning_layers_list = self.flat_layers_list[index:].copy()
            
            
# maybe more to come
