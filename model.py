import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # hidden layer
        self.fc1 = nn.Linear(in_size, hidden_size) # 320, 50
        # output layer
        self.fc2 = nn.Linear(hidden_size, out_size) # 50, 10

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten the image tensors
        x = x.view(-1, 320)
        # Apply activation function
        x = F.relu(self.fc1(x)) 
        x = F.dropout(x, training=self.training)
        # Get predictions using output layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
