import torch.nn as nn
from numpy import sqrt
import torch

class Encoder(nn.Module):
    def __init__(self, vocab_size, dim_size):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim_size)
        self.w_q = nn.Linear(dim_size, dim_size)
        self.w_k = nn.Linear(dim_size, dim_size)
        self.w_v = nn.Linear(dim_size, dim_size)

        # Feed-forward network with expansion
        self.input_proj = nn.Linear(dim_size, (dim_size*4))
        self.relu = nn.ReLU()
        self.output_proj = nn.Linear((dim_size*4), dim_size)
    
    def forward(self, x):
        emb = self.emb(x)  # Embedding the input
        Q = self.w_q(emb)
        K = self.w_k(emb)
        V = self.w_v(emb)

        # Compute scaled dot-product attention
        attention = Q @ K.transpose(-2, -1) / sqrt(64)
        
        # Compute the softmax probabilities
        probs = attention.softmax(dim=-1)
        
        # Weighted sum to get the updated embeddings
        updated = probs @ V

        # Pass through the feed-forward network with expansion
        updated = self.input_proj(updated)
        updated = self.relu(updated)
        updated = self.output_proj(updated)

        return updated


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_size, seq_length):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim_size)
        self.w_q = nn.Linear(dim_size, dim_size)
        self.w_k = nn.Linear(dim_size, dim_size)
        self.w_v = nn.Linear(dim_size, dim_size)

        # Feed-forward network with expansion
        self.input_proj = nn.Linear(dim_size, (dim_size*4))
        self.relu = nn.ReLU()
        self.output_proj = nn.Linear((dim_size*4), dim_size)

        # Final projection to vocabulary size
        self.vocab_proj = nn.Linear(dim_size, vocab_size)
    
    def forward(self, x, encoded_output):
        emb = self.emb(x)  # Embedding the input
        Q = self.w_q(emb)
        K = self.w_k(encoded_output)
        V = self.w_v(encoded_output)
        
        # Compute scaled dot-product attention
        attention = Q @ K.transpose(-2, -1) / sqrt(K.size(-1))
        
        # Create a mask for the upper triangular part (future positions)
        mask = torch.triu(torch.full_like(attention, float("-inf")), diagonal=1)
        
        # Apply the mask to the attention scores
        attn = attention + mask
        
        # Compute the softmax probabilities
        probs = attn.softmax(dim=-1)
        
        # Weighted sum to get the updated embeddings
        updated = probs @ V

        # Pass through the feed-forward network with expansion
        updated = self.input_proj(updated)
        updated = self.relu(updated)
        updated = self.output_proj(updated)

        logits = self.vocab_proj(updated)

        return logits

# Example usage of Encoder and Decoder together
vocab_size = 10
dim_size = 64
x = torch.tensor([1, 2, 3, 4])  # Sample input

encoder = Encoder(vocab_size, dim_size)
decoder = Decoder(vocab_size, dim_size, seq_length=5)

# Pass data through Encoder
encoded_output = encoder(x)  # Shape: [sequence_length, 64]

# Use encoded output in Decoder (assuming x is the input for decoding)
output = decoder(x)
