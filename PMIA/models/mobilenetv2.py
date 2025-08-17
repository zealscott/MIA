import torch
import torch.nn as nn
import os

__all__ = ["MobileNetV2", "create_mobilenetv2"]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, in_channels=3, last_channel=1280, input_size=32, dropout_rate=0.2):
        """
        Args:
            num_classes: number of classes
            width_mult: width multiplier for network
            in_channels: input channels (1 for MNIST, 3 for CIFAR10)
            last_channel: last channel before classifier
            input_size: input image size (28 for MNIST, 32 for CIFAR10)
            dropout_rate: dropout rate for regularization
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        self.input_channel = 32
        self.width_mult = width_mult

        # Adjust initial channel count based on input size
        if input_size == 28:  # MNIST
            self.input_channel = 16  # Smaller initial channels for MNIST
        
        # building first layer
        self.input_channel = int(self.input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        if input_size == 28:  # MNIST - smaller network
            # First conv layer with adaptive in_channels
            layer1 = [
                ConvBNReLU(in_channels, self.input_channel, stride=1),
                self._make_layer(block, 1, 16, 1, 1),  # Reduced channels
                self._make_layer(block, 4, 24, 2, 1),  # Reduced expansion ratio
            ]
            self.layer1 = nn.Sequential(*layer1)

            self.layer2 = self._make_layer(block, 4, 32, 2, 2)  # Reduced layers

            layer3 = [
                self._make_layer(block, 4, 64, 2, 2),  # Reduced layers and channels
                self._make_layer(block, 4, 96, 2, 1),
            ]
            self.layer3 = nn.Sequential(*layer3)

            layer4 = [
                self._make_layer(block, 4, 160, 2, 2),
                self._make_layer(block, 4, 320, 1, 1),
                ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1),
            ]
            self.layer4 = nn.Sequential(*layer4)

        else:  # CIFAR10 - standard network
            # First conv layer with adaptive in_channels
            layer1 = [
                ConvBNReLU(in_channels, self.input_channel, stride=1),
                self._make_layer(block, 1, 16, 1, 1),
                self._make_layer(block, 6, 24, 2, 1),
            ]
            self.layer1 = nn.Sequential(*layer1)

            self.layer2 = self._make_layer(block, 6, 32, 3, 2)

            layer3 = [
                self._make_layer(block, 6, 64, 4, 2),
                self._make_layer(block, 6, 96, 3, 1),
            ]
            self.layer3 = nn.Sequential(*layer3)

            layer4 = [
                self._make_layer(block, 6, 160, 3, 2),
                self._make_layer(block, 6, 320, 1, 1),
                ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1),
            ]
            self.layer4 = nn.Sequential(*layer4)

        # building classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, t, c, n, s):
        output_channel = int(c * self.width_mult)
        layers = []
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(block(self.input_channel, output_channel, stride, expand_ratio=t))
            self.input_channel = output_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x


def create_mobilenetv2(dataset="cifar10", num_classes=10, width_mult=1.0):
    """
    Create MobileNetV2 model for different datasets with appropriate configurations.
    """
    if dataset in ["mnist", "fmnist"]:
        # MNIST configuration - smaller network
        in_channels = 1
        input_size = 28
        width_mult = 0.35  # Reduced network size
        last_channel = 640  # Reduced from 1280
        dropout_rate = 0.2  # Moderate dropout
    else:
        # CIFAR configuration - standard network
        in_channels = 3
        input_size = 32
        width_mult = 1.0  # Standard width
        last_channel = 1280  # Standard size
        dropout_rate = 0.3  # Higher dropout for more regularization

    return MobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        in_channels=in_channels,
        last_channel=last_channel,
        input_size=input_size,
        dropout_rate=dropout_rate
    )
