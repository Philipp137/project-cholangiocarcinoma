from torchvision import transforms, datasets


class ImageDataset(datasets.ImageFolder):
    def __init__(self, data_folder, mode):
        if mode == 'train':
            image_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=[0.8, 1], ratio=[5. / 6., 6. / 5.]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            image_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                 ])
        
        super(ImageDataset, self).__init__(root=data_folder + '/' + mode, transform=image_transforms)
        

