from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the layers
        self.batch_norm_input = nn.BatchNorm2d(3)  # BatchNorm for the input layer

        # First Convolution Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)

        # Second Convolution Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)

        # Third Convolution Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(64)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(64, 3)  # Output 4 classes

    def forward(self, x):
        # Input batch normalization
        x = self.batch_norm_input(x)

        # First Convolution Block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batch_norm1(x)

        # Second Convolution Block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batch_norm2(x)

        # Third Convolution Block
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.batch_norm3(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the dense layer

        # Fully connected layer
        x = self.fc(x)
        
        # Apply softmax to obtain probabilities
        x = F.softmax(x, dim=1)

        return x