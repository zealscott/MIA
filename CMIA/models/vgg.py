import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VGG", "create_vgg16"]


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.classifier = nn.Linear(512, num_classes)
        self.features = self._make_layers(cfg['VGG16'])
        self._initialize_weight()
        
    def forward(self, x):
        # Handle MNIST input: resize to 32x32 and repeat across channels if grayscale
        if x.size(1) == 1:  # If input is grayscale
            x = x.repeat(1, 3, 1, 1)  # Repeat the grayscale channel 3 times
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def create_vgg16(dataset="cifar10", num_classes=10):
    """
    Create VGG16 model for different datasets with appropriate configurations.
    For MNIST/FMNIST, the input will be automatically resized to 32x32 and converted to 3 channels.

    Args:
        dataset: 'mnist', 'fmnist', or 'cifar10'
        num_classes: number of classes

    Returns:
        VGG16 model configured for the specified dataset
    """
    # Always use 3 channels, handle MNIST conversion in forward pass
    model = VGG(num_classes=num_classes, in_channels=3)
    return model