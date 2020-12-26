################################################
# DATA input module
###############################################
# This module implements all routines for
# loading and transforming the data on input
###############################################

from torchvision import transforms, utils, datasets

###############################################
# Definitions of the transformations on input
###############################################
# Data augmentation and normalization for training data
transform_train = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# Only normalization for validation data
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# This is the dictionary which contains both input tranformations
data_transforms = { 'train' : transform_train, 'test': transform_test}