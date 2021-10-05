from torchvision import models
import torch.nn as nn
import timm
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
            
            
class Timm(nn.Sequential):
    def __init__(self, variant='vit_small_patch16_224', dropout=0.0, num_classes=2, activate_logits=False, pretrained=False):
        """
        Build one of the visual transformer variants (or any other model) in timm.list_models()
        """
        self.pretrained = pretrained
        super(Timm, self).__init__()
        self.nn = timm.create_model(variant, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        if activate_logits:
            self.add_module('gelu', nn.GELU())
        
        self.params_base = [p[1] for p in self.nn.named_parameters() if not p[0].startswith('head')]
        self.params_classifier = [p for p in self.nn.head.parameters()]
        self.flat_layers_list = get_layers_list(self.nn)
        self.frozen_layers_list = []
        self.learning_layers_list = self.flat_layers_list.copy()
        
    def freeze_until_nth_layer(self, index):
        for layer in self.flat_layers_list[:index]:
            layer.requires_grad_(False)
            self.frozen_layers_list.append(layer)
        self.learning_layers_list = self.flat_layers_list[index:].copy()
        
