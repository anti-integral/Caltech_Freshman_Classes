import torch
import torch.nn as nn
import numpy as np
from psthree.attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, output_size):
        super(TransformerEncoder, self).__init__()
        
       # TODO: Initialize
       # Ensure you use the norm stated in class and get it from torch.nn
        self.attention = None
        self.norm1 = None
        
        # TODO: Initialize Feed Forward network
        # As described in section 3.3 Attention is all you need paper
        # Note the arguments of this __init__ function 
        # Use nn.Sequential, nn.Linear, and nn.ReLU
        self.ff = None
        self.norm2 = None
        
        # TODO: Initialize output layer
        # Use nn.Linear
        self.output_layer = None
        
    # TODO: Implement forward pass
    def forward(self, x):

        # Self-attention with residual connection and normalization
        x = None
        
        # Feed-forward with residual connection and normalization
        x = None
        
        # Output layer
        output = None
        
        return output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions

class Transformer(nn.Module):
    def __init__(self, tokenizer, embedding_size, num_heads, d_ff, output_size, max_seq_length=512):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.max_seq_length = max_seq_length
        
        # TODO: Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = None
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding())
        
        # TODO: Initialize transformer encoder
        # Use embedding_size for d_model
        self.transformer_encoder = None
    
    def _create_positional_encoding(self):
        position = torch.arange(self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_size, 2) * (-np.log(10000.0) / self.embedding_size))
        pos_encoding = torch.zeros(self.max_seq_length, self.embedding_size)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    # TODO: Implement forward pass
    def forward(self, x):
        # Get sequence length for current batch
        T = x.size(1)
        
        # Token embedding
        x = None
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :T, :]
        
        # Pass through transformer layer
        x = None
        
        return x
