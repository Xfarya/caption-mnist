import torch
import torch.nn as nn
from numpy import sqrt

class Encoder(nn.Module):
    def __init__(self, dim_size):
        super(Encoder, self).__init__()
        
        # Convolutional layers to process the image data (56x56 combined image)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: [32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: [128, 7, 7]
            nn.ReLU(),
        )

        # Fully connected layer to map the flattened conv output to dim_size
        self.fc = nn.Linear(128 * 7 * 7, dim_size)

        # Attention layer components for the encoded representation
        self.w_q = nn.Linear(dim_size, dim_size)
        self.w_k = nn.Linear(dim_size, dim_size)
        self.w_v = nn.Linear(dim_size, dim_size)

        # Feed-forward layers
        self.input_proj = nn.Linear(dim_size, dim_size * 4)
        self.relu = nn.ReLU()
        self.output_proj = nn.Linear(dim_size * 4, dim_size)

    def forward(self, x):
        x = self.conv(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        emb = self.fc(x)  # Get the final encoded representation

        # Attention mechanism
        Q = self.w_q(emb)
        K = self.w_k(emb)
        V = self.w_v(emb)

        # Compute scaled dot-product attention
        attention = Q @ K.transpose(-2, -1) / sqrt(emb.size(-1))
        probs = attention.softmax(dim=-1)
        updated = probs @ V

        # Pass through the feed-forward network
        updated = self.input_proj(updated)
        updated = self.relu(updated)
        updated = self.output_proj(updated)

        return updated


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_size):
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
x = torch.randn(1, 1, 56, 56)  # Simulated combined MNIST image as float tensor


encoder = Encoder(dim_size)
decoder = Decoder(vocab_size, dim_size)

# Pass data through Encoder
encoded_output = encoder(x)  # Shape: [sequence_length, 64]

decoder_input = torch.tensor([[0]], dtype=torch.long)  # <START> token with shape [batch_size, 1]

# Expand `encoded_output` to match the expected input dimensions for the decoder attention
encoded_output = encoded_output.unsqueeze(1)  # Shape: [batch_size, 1, dim_size]

# Pass data through the Decoder
output = decoder(decoder_input, encoded_output)  # Output shape should be [batch_size, 1, vocab_size]

print("Encoder output shape:", encoded_output.shape)  # Expected: [1, 1, dim_size]
print("Decoder output shape:", output.shape)  