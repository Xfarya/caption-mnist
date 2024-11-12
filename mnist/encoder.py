import torch.nn as nn
from numpy import sqrt
import torch

class Encoder(nn.Module):
    def __init__(self, x):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(10000, 64)
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
        
        # Compute the softmax probabilities
        probs = attention.softmax(dim=-1)
        
        # Weighted sum to get the updated embeddings
        updated = probs @ V

        # Debug print statements
        print("Probabilities:\n", probs)