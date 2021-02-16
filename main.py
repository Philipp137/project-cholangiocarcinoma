from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils, datasets,models
from ml4medical import imshow4normalized
import time
import os
import copy
import torch.optim as optim
from torch.optim import lr_scheduler

try:
    import tensorboard
except ImportError as e:
    TB_MODE = False
else:
    TB_MODE = True
    from torch.utils.tensorboard import SummaryWriter
    
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, log_folder="./train_log/"):
    since = time.time()

    tensorboard = SummaryWriter(log_dir=log_folder) if TB_MODE else None
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    steps = dict(train=0, test=0)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if tensorboard and phase == 'train':
                    tensorboard.add_scalar(phase+'_loss', loss.item(), steps[phase])
                steps[phase] += 1
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if tensorboard:
                tensorboard.add_scalar('epoch_' + phase + '_loss', epoch_loss, epoch)
                tensorboard.add_scalar('epoch_' + phase + '_acc', epoch_acc, epoch)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'test':
                save_net_weights(model, log_folder, 'current_wegihts.pt')
                f = open(log_folder + 'train_results.txt', 'a')
                f.write(f"Epoch: {epoch} ;  Loss: {epoch_loss:.3e} ;  Acc: {epoch_acc:.3e} \n")
                f.close()

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_net_weights(model, log_folder, 'best_acc.pt')
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_net_weights(net, fpath, fname):
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    torch.save(net.state_dict(), fpath + fname)

    
if __name__ =="__main__":

    ##### simple example for data loading: #####
    torch.manual_seed(17)
    # first we define the transform done one every image
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=[0.8, 1], ratio=[5./6., 6./5.]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms = { 'train' : transform_train, 'test': transform_test}

    # next we load the dataset
    #data_dir = "./data/project-cholangiocarcinoma/MSIvsMSS/"
    data_dir = "D:/Arbeit/cholangiocarcinoma-data/"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  for x in ['train', 'test']}
    class_names = image_datasets["train"].classes
    # and prepare everything for training
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=192, shuffle=True, num_workers=16)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    ##
    print("Currently we have two classes: %s \ntrain: %d samples\ntest: %d samples\n" % (class_names,dataset_sizes["train"],
                                                                                    dataset_sizes["test"]) )
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))
    # Make a grid from batch
    out = utils.make_grid(inputs)
    imshow4normalized(out, title=[class_names[x] for x in classes])

    # train a model
    model_ft = models.resnet18(num_classes=2, pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)