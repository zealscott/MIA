import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DenseNet121", "create_densenet121"]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0, in_channels=3, input_size=32):
        super(DenseNet121, self).__init__()
        self.num_classes = num_classes
        
        # Adjust initial convolution and pooling based on input size
        if input_size == 28:  # MNIST
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
            )
            self.block_config = [6, 12, 24, 16]  # Standard configuration
        else:  # CIFAR
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
            )
            # Use standard DenseNet121 block configuration
            self.block_config = [6, 12, 24, 16]

        self.num_features = num_init_features
        
        # Dense and transition blocks
        self.dense1 = self._make_dense(self.block_config[0], bn_size, growth_rate, drop_rate)
        self.dense2 = self._make_dense(self.block_config[1], bn_size, growth_rate, drop_rate)
        self.dense3 = self._make_dense(self.block_config[2], bn_size, growth_rate, drop_rate)
        self.dense4 = self._make_dense(self.block_config[3], bn_size, growth_rate, drop_rate, last_dense=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.num_features, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_dense(self, num_layers, bn_size, growth_rate, drop_rate, last_dense=False):
        layers = []
        layers.append(
            _DenseBlock(
                num_layers=num_layers,
                num_input_features=self.num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
        )
        self.num_features = self.num_features + num_layers * growth_rate
        if not last_dense:
            layers.append(_Transition(num_input_features=self.num_features, num_output_features=self.num_features // 2))
            self.num_features = self.num_features // 2
        else:
            layers.append(nn.BatchNorm2d(self.num_features))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_densenet121(dataset="cifar10", num_classes=10):
    """
    Create DenseNet121 model with standard configurations
    """
    if dataset in ["mnist", "fmnist"]:
        in_channels = 1
        input_size = 28
        growth_rate = 32  # Keep standard growth rate
        num_init_features = 64  # Keep standard initial features
        bn_size = 4
        drop_rate = 0.2
    else:  # CIFAR standard parameters
        in_channels = 3
        input_size = 32
        growth_rate = 32  # Standard growth rate
        num_init_features = 64  # Standard initial features
        bn_size = 4
        drop_rate = 0.2

    return DenseNet121(
        num_classes=num_classes,
        growth_rate=growth_rate,
        num_init_features=num_init_features,
        bn_size=bn_size,
        drop_rate=drop_rate,
        in_channels=in_channels,
        input_size=input_size
    )
