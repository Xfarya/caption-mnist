import torch
import torch.nn as nn
from numpy import sqrt

class Decoder(nn.Module):
    def __init__(self, x):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(5, 64)
        self.w_q = nn.Linear(64, 64)
        self.w_k = nn.Linear(64, 64)
        self.w_v = nn.Linear(64, 64)
    
    def forward(self, x):
        emb = self.emb(x)  # Embedding the input
        Q = self.w_q(emb)
        K = self.w_k(emb)
        V = self.w_v(emb)
        
        # Compute scaled dot-product attention
        attention = Q @ K.transpose(-2, -1) / sqrt(64)
        
        # Create a mask for the upper triangular part (future positions)
        mask = torch.triu(torch.full_like(attention, float("-inf")), diagonal=1)
        
        # Apply the mask to the attention scores
        attn = attention + mask
        
        # Compute the softmax probabilities
        probs = attn.softmax(dim=-1)
        
        # Weighted sum to get the updated embeddings
        updated = probs @ V

        # Debug print statements
        print("Mask:\n", mask)
        print("Probabilities:\n", probs)

        return updated

# Example usage
x = torch.tensor([1, 2, 3, 4])  # Sample input
decoder = Decoder(x)
output = decoder(x)
