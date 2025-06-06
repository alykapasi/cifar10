import json
import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 conv1_channels: int = 32,
                 conv2_channels: int = 64,
                 conv3_channels: int = 128,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.25,
                 fc1_size: int = 256,
                 fc2_size: int = 128
                 ):
        super(BasicCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, kernel_size, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(conv3_channels * 4 * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def load_config(config_path):
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_model(config_path):
    """Create BasicCNN model from config file"""
    config = load_config(config_path)
    return BasicCNN(config)