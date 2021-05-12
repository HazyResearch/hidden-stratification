from collections import OrderedDict

import torch.nn as nn


class ShallowCNN(nn.Module):
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
                    ('c3', nn.Conv2d(6, 8, kernel_size=(5, 5))),
                    ('relu3', nn.ReLU()),
                ]
            )
        )

        # Store the name of the layer whose output we want to use as features
        # (typically the last operation before the classification layer)
        self.activation_layer_name = 'convnet.relu3'
        self.fc = nn.Linear(512, classes)

    def forward(self, img, save_act=None):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


def shallow_cnn(**kwargs):
    model = ShallowCNN(**kwargs)
    return model
