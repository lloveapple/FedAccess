import torch.nn as nn
import torchvision.models


class ResNet(nn.Module):
    def __init__(self, hyperparameters):
        super(ResNet, self).__init__()
        if hyperparameters.get("resnet18", False):
            self.network = torchvision.models.resnet18()
            self.network.fc = nn.Linear(self.network.fc.in_features, hyperparameters.get("class_num"))
        else:
            self.network = torchvision.models.resnet50()
            self.network.fc = nn.Linear(self.network.fc.in_features, hyperparameters.get("class_num"))
