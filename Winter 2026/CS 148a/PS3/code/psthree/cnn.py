import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=28):  # Default to MNIST settings
        super(CNN, self).__init__()
        # TODO: Initialize layers
        # Use nn.Conv2d for the convolutional layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # Use 4 filters with a filter size of 3 with padding 1 and stride 1
        # Note that the input images are grayscale
        # Use nn.Linear for the fully connected layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.conv = None
        self.fc = None
    
    # TODO: Implement forward pass
    def forward(self, x):
        # pass through conv layer
        x = None

        # apply relu activation
        # Use F.relu
        x = None

        # flatten the output
        # Use torch.flatten
        x = None

        # pass through fully connected layer
        x = None

        return x

class ManualCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(ManualCNN, self).__init__()
        # TODO: Initialize layers same as before
        self.conv = None
        self.fc = None

        # Set filters manually
        self.conv.weight.data = torch.tensor([
            # Filter 1
            [[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]],
            # Filter 2
            [[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]],
            # Filter 3
            [[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]],
            # Filter 4
            [[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]],
        ], dtype=torch.float32)

        # Ensure that the conv weights and biases are not trainable                                        
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    # TODO: Implement forward pass
    def forward(self, x):
        # pass through conv layer
        x = None

        # apply relu activation
        # Use F.relu
        x = None

        # flatten the output
        # Use torch.flatten
        x = None

        # pass through fully connected layer
        x = None

        return x