import torch
import torch.nn as nn
import numpy as np

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # TODO: Initialize weights and biases 
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation
        self.W_xh = None
        self.W_hh = None
        self.b_h = None
        self.W_hy = None
        self.b_y = None
        
        # TODO: Initialize activation function
        # Use nn.Tanh
        self.tanh = None

    # TODO: Implement forward pass
    def forward(self, x):
        
        # Define dimensions
        B = None
        T = None
        H = None
        
        # Initialize hidden state to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        hidden = None
        
        # List of outputs for each time step
        outputs = []
        
        # Calculate hidden and output for each time step and update variables
        # Follow the equations in the problem set
        # torch.matmul is used for matrix multiplication
        # ----------------------------------------

        # ----------------------------------------
        
        # Stack outputs along the time dimension to get a tensor of shape (B, T, output_size)
        # Hint: use torch.stack
        outputs = None
        
        return outputs

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions

class RNN(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = None
        self.rnn_layer = None
    
    # TODO: Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = None

        # Pass through rnn layer
        x = None

        return x