import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import wandb

from encoderdecoder import Encoder, Decoder

# Define the Combine dataset
class Combine(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.tf = torchvision.transforms.ToTensor()
        self.ds = torchvision.datasets.MNIST(root='.', download=True, transform=self.tf)
        self.ln = len(self.ds)

    def __len__(self):
        return self.ln

    def __getitem__(self, idx):
        idx = random.sample(range(self.ln), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)

        img = torch.zeros((1, 56, 56))
        img[:, 0:28, 0:28] = store[0]
        img[:, 0:28, 28:56] = store[1]
        img[:, 28:56, 0:28] = store[2]
        img[:, 28:56, 28:56] = store[3]

        label = torch.tensor(label, dtype=torch.long)  # Convert label list to tensor

        return img, label

# Model parameters
vocab_size = 10
dim_size = 64
batch_size = 8
num_epochs = 10
sequence_length = 4
num_heads = 8  # Number of attention heads

# Initialize W&B project
wandb.init(project="caption-mnist", name="multi-head-attention-test")

wandb.config = {
    "epochs": num_epochs,
    "batch_size": batch_size,
    "vocab_size": vocab_size,
    "dim_size": dim_size,
    "num_heads": num_heads,
    "learning_rate": 0.001,
}

# Initialize Encoder and Decoder with multi-head attention
encoder = Encoder(dim_size, num_heads)
decoder = Decoder(vocab_size, dim_size, num_heads)

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

# Initialize the Combine dataset and DataLoader with default collate_fn
ds = Combine()
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=0.001
)

# Training loop for multiple batches
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    max_batches = 100  # Limit the number of batches per epoch for faster testing

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        optimizer.zero_grad()

        # Move data to the appropriate device
        images = images.to(device)
        labels = labels.to(device)

        # Encode the batch of combined images
        encoder_output = encoder(images)  # Shape: [batch_size, encoder_seq_len, dim_size]

        # Prepare decoder input (<START> tokens)
        decoder_input = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)

        # Prepare targets
        # labels: [batch_size, sequence_length]
        targets = labels  # Shape: [batch_size, sequence_length]

        # Pass through the decoder
        output = decoder(decoder_input, encoder_output)  # Output shape: [batch_size, sequence_length, vocab_size]

        # Compute loss
        output_logits = output.view(-1, vocab_size)  # Shape: [batch_size * sequence_length, vocab_size]
        flattened_targets = targets.contiguous().view(-1)  # Shape: [batch_size * sequence_length]

        loss = criterion(output_logits, flattened_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log batch loss to W&B
        wandb.log({
            "batch_loss": loss.item(),
            "batch_idx": batch_idx + 1,
            "epoch": epoch + 1
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{max_batches}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / (batch_idx + 1)

    wandb.log({
        "epoch_loss": avg_loss,
        "epoch": epoch + 1
    })
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

wandb.finish()
