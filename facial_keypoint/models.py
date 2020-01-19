## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.pool3 = nn.MaxPool2d(2, 2) 
        self.pool4 = nn.MaxPool2d(2, 2) 
        
        self.flatten = Flatten()

        self.fc1 = nn.Linear(6400,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,136)
        
        
    def forward(self, x):

        x = self.dropout1(self.pool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.pool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.pool4(F.elu(self.conv4(x))))
        
        x = self.dropout5(F.elu(self.fc1(self.flatten(x))))
        x = self.fc3(self.dropout6(F.elu(self.fc2(x))))
        
        return x
