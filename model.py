# Model.py
import torchvision.models as models
import torch
import torch.nn as nn

class Resnet50(nn.Module):
    '''
    Resnet 50.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128):
        super(Resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(2048, dim)
        
    def forward(self, x):
        out = self.resnet(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        return out / norm