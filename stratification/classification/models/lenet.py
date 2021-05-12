from collections import OrderedDict

import torch.nn as nn


class LeNet4(nn.Module):
    """
    Adapted from https://github.com/activatedgeek/LeNet-5
    """

    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('num_channels', 1)
        classes = kwargs.get('num_classes', 10)
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ('c1', nn.Conv2d(in_channels, 6, kernel_size=(5, 5))),
                    ('relu1', nn.ReLU()),
                    ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
                    ('relu3', nn.ReLU()),
                    ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
                    ('relu5', nn.ReLU()),
                ]
            )
        )

        self.activation_layer_name = 'convnet.relu5'
        self.fc = nn.Linear(120, classes)

    def forward(self, img):
        x = self.convnet(img)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
