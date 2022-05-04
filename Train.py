from tqdm.notebook import tqdm_notebook
from CIFAR10 import LoadDataset
from ResNet import ResNet
from TrainingFunctions import Network
from Augmentations import GetAugment
from Plots import plot

# Parameters
epochs = 40
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-3
batch = 256
n = 3

""" Configurations is a list of plain(normalization), baseline, autoaugment augementations. """
configurations = GetAugment()
names = ["Plain", "Baseline", "AutoAugment"]
normalization = configurations[0]


""" Accuracies and Losses for each model. """
accuracy = []
loss = []

""" Traning model for each augmentation -> Plain, Baseline, RandAug. """
for i in range(len(configurations)):
    model = ResNet(n).to(device)
    train_loader, val_loader, test_loader = LoadDataset(batch, normalization, configurations[i])
    network = Network(model, learning_rate, device)
    for epoch in tqdm_notebook(range(epochs), desc='Epoch'):
        network.train_step(train_loader)
        network.validation_step(val_loader)
        print("{} => ".format(names[i])  + "Train acc : {}".format(network.train_acc[-1]), "Val acc : {}".format(network.val_acc[-1]))
    network.test_step(test_loader)
    network.save_all(names[i])
    accuracy.append([network.train_acc, network.val_acc])
    loss.append([network.train_loss, network.val_loss])

plot(accuracy, "Accuracy")
plot(loss, "Loss")
