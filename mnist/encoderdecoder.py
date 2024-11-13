import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_size, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, dim_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim_size, 2).float() * (-math.log(10000.0) / dim_size))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, dim_size]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim_size, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )
        self.norm1 = nn.LayerNorm(dim_size)
        self.norm2 = nn.LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    def __init__(self, dim_size, num_heads, num_layers=3, dropout=0.1):
        super(Encoder, self).__init__()
        
        # Convolutional layers to process image data
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2d(64, dim_size, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, dim_size, 7, 7]
            nn.ReLU(),
        )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(dim_size, max_len=7*7, dropout=dropout)
        
        # Stacked Encoder Layers
        self.layers = nn.ModuleList([EncoderLayer(dim_size, num_heads, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        # x: [batch_size, 1, 56, 56]
        x = self.conv(x)  # [batch_size, dim_size, 7, 7]
        batch_size, dim_size, H, W = x.shape
        x = x.view(batch_size, dim_size, H * W).transpose(1, 2)  # Reshape to [batch_size, sequence_length, dim_size]
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)
            
        return x  # [batch_size, sequence_length, dim_size]

class DecoderLayer(nn.Module):
    def __init__(self, dim_size, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_size, dim_size * 4),
            nn.ReLU(),
            nn.Linear(dim_size * 4, dim_size),
        )
        self.norm1 = nn.LayerNorm(dim_size)
        self.norm2 = nn.LayerNorm(dim_size)
        self.norm3 = nn.LayerNorm(dim_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None):
        # Masked self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer normalization
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_size, num_heads, num_layers=3, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.positional_encoding = PositionalEncoding(dim_size, max_len=100, dropout=dropout)
        
        # Stacked Decoder Layers
        self.layers = nn.ModuleList([DecoderLayer(dim_size, num_heads, dropout) for _ in range(num_layers)])
        
        # Final projection to vocabulary size
        self.fc_out = nn.Linear(dim_size, vocab_size)
        
    def forward(self, x, encoder_output):
        # x: [batch_size, sequence_length]
        x = self.embedding(x)  # [batch_size, sequence_length, dim_size]
        x = self.positional_encoding(x)
        
        # Generate mask to prevent attention to future positions
        seq_len = x.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)
        
        logits = self.fc_out(x)  # [batch_size, sequence_length, vocab_size]
        return logits
        
    def generate_square_subsequent_mask(self, sz):
        # Create mask for the sequence
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
