import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.Conv2d(8, 16, kernel_size=4)
        )
    
    def forward(self, x):
        return self.net(x)

model = Net()
print(model(torch.zeros((1,3,6,6))).shape)