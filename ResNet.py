import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    """ ConvBlock consists of Conv2d, BatchNorm, and activation function set as default to ReLU. """

    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        padding : int,
        act = nn.ReLU()
        ):
        super().__init__()

        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))

class ResidualBlock(nn.Module):
    """ Bottleneck ResidualBlock that takes in_channels, out_channels and information if it's first block of the network. """

    def __init__(
        self,
        in_channels,
        out_channels,
        first=False
        ):
        super().__init__()

        """ Projection for residual connection when the number of channels is increasing. """
        self.projection = in_channels != out_channels
        self.relu = nn.ReLU(inplace=True)
        stride = 1

        if self.projection:
            stride = 2
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

        if first:
            stride = 1
            self.p = ConvBlock(in_channels, in_channels, 1, stride, 0)

        self.c1 = ConvBlock(in_channels, in_channels, 3, 1, 1)
        self.c2 = ConvBlock(in_channels, out_channels, 3, stride, 1, nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        f = self.c1(x)
        f = self.c2(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h

# ResNetx
class ResNet(nn.Module):
    """ ResNetX architecture, where X is the number of layers and depends on configuration(n - number of blocks at each stage) we pass as input. """

    def __init__(
        self,
        n : int,
        in_channels=3,
        classes=10
        ):
        super().__init__()

        out_features = [16, 32, 64]
        no_blocks = [n] * 3

        """ Adding first block of the network manually. """
        self.blocks = nn.ModuleList([ResidualBlock(16, 16, True)])

        """ Adding all residual blocks using loop. """
        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i]))
            for _ in range(no_blocks[i]-1):
                self.blocks.append(ResidualBlock(out_features[i], out_features[i]))

        """ Final stage of the network with Conv2d -> Global Pooling -> Fully-connected layers. """
        self.conv1 = ConvBlock(in_channels, 16, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, classes)

        self.init_weight()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    """ Kaiming He weight initialization. """
    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
