# Data Augmentation in Torchvison
This is an analysis of different data augmentations techniques in Torchvision evaluated on CIFAR10. You can find the accompanying article on my [__Medium__](https://medium.com/@maciejbalawejder) page.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/marvin.png"
>
</p>

# Table of content: 

### Augmentations
  [Augmentations.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/Augmentations.py) - three transformation setups:

  1. Plain - only `Normalize()` operation.

  2. Baseline - `HorizontalFlipping()`, `RandomCrop()`, `RandomErasing()`.

  3. AutoAugment - `AutoAugment` policy for CIFAR10 applied on the top of Baseline configuration.

  ```
  from Augmentations import GetAugment
  plain, baseline, autoaugment = GetAugment()
  ```
  
### Dataset

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/CIFAR10.png"
>
</p>

[CIFAR10.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/CIFAR10.py) - file with CIFAR-10 loader which takes __batch size__, __normalization__ for train and validation set and __augmentations__ for training.

```
from CIFAR10 import LoadDataset
trainloader, valloader, testloader = LoadDataset(batch, normalization, augmentations) 
```

### Model

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/ResNet.png"
>
</p>

[ResNet.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/ResNet.py) - ResNet configuration specially tailored for CIFAR-10 dataset, which takes parameter `n` that defines the number of `residual blocks` at each stage.

```
from ResNet import ResNet
n = 3
resnet20 = ResNet(n)
```

### Training Loops
[TrainingFunctions.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/TrainingFunctions.py) - file with training, validation, testing loop.

```
from TrainingFunctions import Network
network = Network(model=ResNet(3), learning_rate=0.01, device="cuda")
network.train_step(trainloader)
```

### Plots

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/Loss.png"
>
</p>

[Plots.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/Plots.py) - generates one plot for multiple models.

```
from Plots import plot
plot([model1_train_loss, model1_val_loss, model2_train_loss, model2_val_los], "Loss")
```

### Training
[Train.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/Train.py) - combines all of the files above and train three different configurations
