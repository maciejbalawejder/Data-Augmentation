# Data Augmentation in Torchvison
This is an analysis of different data augmentations techniques in Torchvision evaluated on CIFAR10. You can find the accompanying article on my [__Medium__](https://medium.com/@maciejbalawejder) page.

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/marvin.png"
>
</p>

# Table of content: 

### Augmentations

  1. Plain - only `Normalize()` operation.

  2. Baseline - `HorizontalFlipping()`, `RandomCrop()`, `RandomErasing()`.

  3. AutoAugment - `AutoAugment` policy for CIFAR10 applied on the top of Baseline configuration.

  ```python
  from augmentations import GetAugment
  plain, baseline, autoaugment = GetAugment()
  ```
  
### Dataset

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/CIFAR10.png"
>
</p>

```python
from cifar10 import LoadDataset
trainloader, valloader, testloader = LoadDataset(batch, normalization, augmentations) 
```

### Model

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/ResNet.png"
>
</p>

```python
from resnet import ResNet
n = 3
resnet20 = ResNet(n)
```

### Training Loops

```python
from training_functions import Network
network = Network(model=ResNet(3), learning_rate=0.01, device="cuda")
network.train_step(trainloader)
```

### Plots

<p align="center">
<img 
  src="https://github.com/maciejbalawejder/Data-Augmentation/blob/main/images/Loss.png"
>
</p>

```python
from plots import plot
plot([model1_train_loss, model1_val_loss, model2_train_loss, model2_val_los], "Loss")
```

### Training
[train.py](https://github.com/maciejbalawejder/Data-Augmentation/blob/main/Train.py) - combines all of the files above and train three different configurations
