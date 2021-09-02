import torch
from torchvision import datasets, transforms

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
def input_normalize(x):
    _normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    return _normalize(x)
def invert_normalize(x):
    _inv_normalize = transforms.Normalize(
        mean= [-m/s for m, s in zip(imagenet_mean, imagenet_std)],
        std= [1/s for s in imagenet_std])
    return _inv_normalize(x)

def get_loaders(data_directory, batch_size, image_size, testonly=False): 
    # only support imagenet-size image (224x224)
    # move normalize into model, don't normalize here, 
    # is better for classic adversarial attacks
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std), 
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
    if testonly:
        test_dataset = datasets.ImageFolder(root=data_directory, \
        transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
            shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
        return test_loader
    
    print('==> Preparing dataset..')
    train_dataset = datasets.ImageFolder(root=data_directory+'/train', \
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=data_directory+'/val', \
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,\
        shuffle=True, drop_last=True, num_workers=12, pin_memory=True)
    return train_loader, test_loader