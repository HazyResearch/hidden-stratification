import torch.nn as nn
import torch.nn.functional as F


class LeNet300100(nn.Module):
    def __init__(self, **kwargs):
        super(LeNet300100, self).__init__()
        linear = nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)
        self.activation_layer_name = 'fc2'

    def forward(self, x, save_act=False):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), -1)
        return x


def lenet300100(**kwargs):
    model = LeNet300100(**kwargs)
    return model
