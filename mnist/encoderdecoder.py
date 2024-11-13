import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, dim_size, num_heads):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [batch_size, 32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch_size, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, dim_size, kernel_size=3, stride=2, padding=1),  # [batch_size, dim_size, 7, 7]
            nn.ReLU(),
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(dim_size, max_len=7*7)

        # Multi-head Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_size, num_heads=num_heads, batch_first=True)

        # Feed-forward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )
        
    def forward(self, x):
        # x: [batch_size, 1, 56, 56]
        x = self.conv(x)  # x: [batch_size, dim_size, 7, 7]
        batch_size, dim_size, H, W = x.shape
        x = x.view(batch_size, dim_size, -1).transpose(1, 2)  # x: [batch_size, sequence_length, dim_size]
        # sequence_length = H * W = 49

        # Add positional encoding
        x = self.positional_encoding(x)

        # Self-attention
        x_attn, _ = self.self_attn(x, x, x)  # x_attn: [batch_size, sequence_length, dim_size]

        # Residual connection and feed-forward network
        x = x + x_attn  # Residual connection
        x = self.feed_forward(x)  # x: [batch_size, sequence_length, dim_size]

        return x  # x: [batch_size, sequence_length, dim_size]


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_size, num_heads):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim_size)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(dim_size, max_len=100)

        # Multi-head Attention layers
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_size, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_size, num_heads=num_heads, batch_first=True)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )

        # Final projection to vocabulary size
        self.vocab_proj = nn.Linear(dim_size, vocab_size)
        
    def forward(self, x, encoder_output):
        # x: [batch_size, sequence_length]
        # encoder_output: [batch_size, encoder_sequence_length, dim_size]

        # Embedding and positional encoding
        x = self.emb(x)  # x: [batch_size, sequence_length, dim_size]
        x = self.positional_encoding(x)

        # Masked self-attention (causal)
        tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)  # x.size(1) is sequence_length
        x_attn, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = x + x_attn  # Residual connection

        # Cross-attention with encoder outputs
        x_attn, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = x + x_attn  # Residual connection

        # Feed-forward network
        x = self.feed_forward(x)

        # Project to vocabulary size
        logits = self.vocab_proj(x)  # logits: [batch_size, sequence_length, vocab_size]

        return logits  # [batch_size, sequence_length, vocab_size]

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. Mask out subsequent positions."""
        mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, dim_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim_size, 2).float() * (-math.log(10000.0) / dim_size))
        pe = torch.zeros(max_len, dim_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, dim_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, sequence_length, dim_size]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
