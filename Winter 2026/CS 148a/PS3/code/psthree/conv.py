import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1, filters=None, biases=None):
        super(Conv2d, self).__init__()

        # Number of input channels
        self.in_channels = in_channels

        # Number of filters
        self.out_channels = out_channels

        # Width/height of filters
        self.filter_size = filter_size

        # Padding for width and height
        self.padding = padding

        # Stride for width and height
        self.stride = stride

        self.filters = nn.Parameter(filters if filters is not None else torch.randn(out_channels, in_channels, filter_size, filter_size))
        self.biases = nn.Parameter(biases if biases is not None else torch.randn(out_channels))

    # TODO: Complete this function
    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """

        # Batch size
        B = None

        # Number of input channels
        C_in = None

        # Height of input
        H = None

        # Width of input
        W = None

        # Number of output channels
        C_out = None

        # Width/height of filter
        F = None

        # Padding
        P = None
        
        # Stride
        S = None

        # Compute height of output considering padding and stride
        # Account for both even and odd dimensions
        H_out = None

        # Compute width of output considering padding and stride
        # Account for both even and odd dimensions
        W_out = None

        # Initialize output tensor with zeros
        # Use torch.zeros
        output = None

        # Compute padded_x to be x with P zero padding on all sides
        # Hint: Use torch.zeros and then set the appropriate elements to those of x
        padded_x = None

        # Loop over all necessary dimensions to fill in the output tensor
        # To do this, you need to perform a 2d convolution of each filter over padded_x
        # Don't forget to account for the stride
        # Hint: You'll have 4 nested for loops
        # ----------------------------------------

        # ----------------------------------------

        return output
        

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)

class FasterConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, padding=0, stride=1, filters=None, biases=None):
        super(FasterConv2d, self).__init__()

         # Number of input channels
        self.in_channels = in_channels

        # Number of filters
        self.out_channels = out_channels

        # Width/height of filters
        self.filter_size = filter_size

        # Padding for width and height
        self.padding = padding

        # Stride for width and height
        self.stride = stride
        
        self.filters = nn.Parameter(filters if filters is not None else torch.randn(out_channels, in_channels, filter_size, filter_size))
        self.biases = nn.Parameter(biases if biases is not None else torch.randn(out_channels))

    # TODO: Complete this function
    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """

         # Batch size
        B = None

        # Number of input channels
        C_in = None

        # Height of input
        H = None

        # Width of input
        W = None

        # Number of output channels
        C_out = None

        # Width/height of filter
        F = None

        # Padding
        P = None
        
        # Stride
        S = None

        # Compute height of output considering padding and stride
        # Account for both even and odd dimensions
        H_out = None

        # Compute width of output considering padding and stride
        # Account for both even and odd dimensions
        W_out = None

        # Initialize output tensor with zeros
        # Use torch.zeros
        output = None

        # Compute padded_x to be x with P zero padding on all sides
        # Hint: Use torch.zeros and then set the appropriate elements to those of x
        padded_x = None

        # If you think 4 nested for loops is computationally inefficient, you're right!
        # Let's do it faster, using broadcasting
        # If you've never heard of broadcasting, here's a good article to get acquainted: https://www.geeksforgeeks.org/understanding-broadcasting-in-pytorch/
        # For this implementation, you should only loop over H_out and W_out
        # Then, you'll need to update the shape of filters and the current patch from padded_x
        # so that you multiply every patch in the batch by every filter in filters
        # Ensure that you fill in the element of output accordingly
        # ----------------------------------------

        # ----------------------------------------

        return output

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)
    

