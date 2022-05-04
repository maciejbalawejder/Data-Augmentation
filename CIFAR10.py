import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# CIFAR10
def LoadDataset(batch, normalization, augmentation):
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=augmentation)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch,
                                              shuffle=True,
                                              num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=normalization)

    testset, valset = torch.utils.data.random_split(testset, [7500, 2500])

    valloader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch,
                                             shuffle=False,
                                             num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch,
                                             shuffle=False,
                                             num_workers=2)

    return trainloader, valloader, testloader
