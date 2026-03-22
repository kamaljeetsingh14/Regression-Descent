# src/models/small_cnn.py
import torch.nn as nn
import torch.nn.functional as F

    
class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)    # Output: (8, 26, 26)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   # Output: (16, 24, 24)
        self.pool = nn.MaxPool2d(2, 2)                 # After pool: (16, 12, 12)
        self.fc1 = nn.Linear(16 * 12 * 12, 128)        # First FC layer
        self.fc2 = nn.Linear(128, 10)                  # Output layer (e.g., for 10 classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Conv1 + ReLU
        x = F.relu(self.conv2(x))      # Conv2 + ReLU
        x = self.pool(x)               # Max pooling
        x = x.view(-1, 16 * 12 * 12)   # Flatten
        x = F.relu(self.fc1(x))        # FC1 + ReLU
        x = self.fc2(x)                # Output layer
        return x

class MediumCNN(nn.Module):
    def __init__(self):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)   # 1 -> 16 channels
        self.pool = nn.MaxPool2d(2, 2)                 # 28x28 -> 13x13 (after two convs and 1 pool)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 16 -> 32 channels
        self.fc1 = nn.Linear(32 * 11 * 11, 64)         # add a hidden FC layer
        self.fc2 = nn.Linear(64, 10)                   # output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))                      # conv1 + ReLU
        x = self.pool(F.relu(self.conv2(x)))           # conv2 + ReLU + pool
        x = x.view(-1, 32 * 11 * 11)                   # flatten
        x = F.relu(self.fc1(x))                        # hidden FC + ReLU
        x = self.fc2(x)                                # output logits
        return x
    


class WiderCNN(nn.Module):
    def __init__(self):
        super(WiderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)    # Output: (16, 26, 26)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)   # Output: (32, 24, 24)
        self.pool = nn.MaxPool2d(2, 2)                  # After pool: (32, 12, 12)
        self.fc1 = nn.Linear(32 * 12 * 12, 256)         # Wider hidden layer
        self.fc2 = nn.Linear(256, 10)                   # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 1 input channel, 8 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)               # 2x2 max pooling
        self.fc1 = nn.Linear(8 * 13 * 13, 10)        # after conv+pool, size is 13x13

    def forward(self, x):
        x = F.relu(self.conv1(x))   # conv + ReLU
        x = self.pool(x)            # max pooling
        x = x.view(-1, 8 * 13 * 13) # flatten
        x = self.fc1(x)             # fully connected layer
        return x
        


class BigMNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # Ensure output 1x1 for fc
        
        # Fully connected layers
        self.fc1 = nn.Linear(512*1*1, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Convolution + pooling
        x = F.relu(self.conv1(x))          # 28x28
        x = self.pool(F.relu(self.conv2(x)))  # 14x14
        x = self.pool(F.relu(self.conv3(x)))  # 7x7
        x = self.pool(F.relu(self.conv4(x)))  # 3x3
        
        # Adaptive pooling to 1x1
        x = self.global_pool(x)              # 1x1
        x = torch.flatten(x, 1)              # flatten to [batch, 512]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))              # [batch, 1024]
        x = self.fc2(x)                      # [batch, num_classes]
        return x

class SlightlyDifficultCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)     # (28x28) → (24x24)
        self.pool = nn.MaxPool2d(2, 2)                  # 2x2 max pooling
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)    # (12x12) → (8x8)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)            # Flattened after pool
        self.fc2 = nn.Linear(64, 10)                    # Final output

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (28,28) → (24,24)
        x = self.pool(x)               # (24,24) → (12,12)
        x = F.relu(self.conv2(x))      # (12,12) → (8,8)
        x = self.pool(x)               # (8,8) → (4,4)
        x = x.view(-1, 16 * 4 * 4)     # Flatten to (batch, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x     
    
# Example usage:

class Bottleneck(nn.Module):
    expansion = 4  # Output channel multiplier

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

