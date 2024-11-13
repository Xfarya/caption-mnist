import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import wandb

from encoderdecoder import Encoder, Decoder

class CombineDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()
        self.dataset = torchvision.datasets.MNIST(root='.', download=True, transform=self.transform)
        self.length = len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        indices = random.sample(range(self.length), 4)
        images = []
        labels = []
        
        for i in indices:
            image, label = self.dataset[i]
            images.append(image)
            labels.append(label)
        
        # Combine four images into one 56x56 image
        combined_image = torch.zeros((1, 56, 56))
        combined_image[:, 0:28, 0:28] = images[0]
        combined_image[:, 0:28, 28:56] = images[1]
        combined_image[:, 28:56, 0:28] = images[2]
        combined_image[:, 28:56, 28:56] = images[3]
        
        labels = torch.tensor(labels, dtype=torch.long)  # [sequence_length]
        return combined_image, labels  # [1, 56, 56], [4]

# Model parameters
vocab_size = 10
dim_size = 64
batch_size = 8
num_epochs = 10
sequence_length = 4
num_heads = 8
num_layers = 3
learning_rate = 0.001
max_batches = 100

# Initialize W&B project
wandb.init(project="caption-mnist", name="transformer-mnist")

wandb.config = {
    "epochs": num_epochs,
    "batch_size": batch_size,
    "vocab_size": vocab_size,
    "dim_size": dim_size,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "learning_rate": learning_rate,
    "max_batches": max_batches
}

# Initialize models
encoder = Encoder(dim_size, num_heads, num_layers)
decoder = Decoder(vocab_size, dim_size, num_heads, num_layers)

# Move models to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

# Prepare dataset and dataloader
dataset = CombineDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
)

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
                break 
        optimizer.zero_grad()
        
        # Move data to device
        images = images.to(device)  # [batch_size, 1, 56, 56]
        labels = labels.to(device)  # [batch_size, sequence_length]
        
        # Forward pass through encoder
        encoder_output = encoder(images)  # [batch_size, encoder_seq_len, dim_size]
        
        # Prepare decoder input (<START> tokens)
        decoder_input = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
        
        # Forward pass through decoder
        output = decoder(decoder_input, encoder_output)  # [batch_size, sequence_length, vocab_size]
        
        # Compute loss
        loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch loss
        wandb.log({
            "batch_loss": loss.item(),
            "epoch": epoch + 1
        })
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    wandb.log({
        "epoch_loss": avg_loss,
        "epoch": epoch + 1
    })
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

wandb.finish()
