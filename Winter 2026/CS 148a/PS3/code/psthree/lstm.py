import torch
import torch.nn as nn
import numpy as np

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # TODO: Initialize weights and biases 
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation

        # Input gate weights
        self.W_xi = None
        self.W_hi = None
        self.b_i = None
        
        # Forget gate weights
        self.W_xf = None
        self.W_hf = None
        self.b_f = None
        
        # Output gate weights
        self.W_xo = None
        self.W_ho = None
        self.b_o = None
        
        # Cell state weights
        self.W_xc = None
        self.W_hc = None
        self.b_c = None
        
        # Output weights
        self.W_hy = None
        self.b_y = None
        
        # TODO: Initialize activation functions
        # Use nn.Tanh, nn.Sigmoid
        self.sigmoid = None
        self.tanh = None


    # TODO: Implement forward pass
    def forward(self, x):
        
        # Define dimensions
        B = None
        T = None
        H = None
        
        # Initialize h_t and c_t to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        h_t = None
        c_t = None
        
        # List of outputs for each time step
        outputs = []
        
        # Calculate h_t, c_t, and output for each time step and update variables
        # Follow the equations in the problem set
        # Element-wise multiplcation can be done with just "*"
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

class LSTM(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = None
        self.lstm_layer = None
    
    # TODO: Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = None

        # Pass through lstm layer
        x = None

        return x