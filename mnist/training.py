import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import wandb

from encoderdecoder import Decoder, Encoder

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

# Initialize W&B project
wandb.init(project="caption-mnist", name="multi-batch-test-debug")

wandb.config = {
    "epochs": num_epochs,
    "batch_size": batch_size,
    "vocab_size": vocab_size,
    "dim_size": dim_size,
    "learning_rate": 0.001,
}

# Initialize Encoder and Decoder
encoder = Encoder(dim_size)
decoder = Decoder(vocab_size, dim_size)

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

    max_batches = 100

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")

        if batch_idx >= max_batches:
            break

        optimizer.zero_grad()

        # Encode the batch of combined images
        encoded_output = encoder(images)  # Shape: [batch_size, dim_size]
        print(f"Encoded output shape: {encoded_output.shape}")  # Expected: [batch_size, dim_size]

        # Ensure labels are tensors with the correct shape
        print(f"Labels shape: {labels.shape}")  # Expected: [batch_size, sequence_length]

        # Initialize lists to collect logits and targets
        batch_logits = []
        batch_targets = []

        # Process each sample in the batch
        for i in range(batch_size):
            single_encoded_output = encoded_output[i].unsqueeze(0)  # Shape: [1, dim_size]
            single_label = labels[i]  # Tensor of shape [sequence_length]

            decoder_input = torch.zeros((1, 1), dtype=torch.long, device=single_encoded_output.device)

            # Collect logits for this sequence
            sample_logits = []

            for j in range(sequence_length):
                logits = decoder(decoder_input, single_encoded_output)  # Shape: [1, 1, vocab_size]
                sample_logits.append(logits.squeeze(1))  # Shape: [1, vocab_size]

                # Update decoder input with the next token
                decoder_input = single_label[j].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]

            # Stack and store logits and labels for this sample
            sample_logits = torch.cat(sample_logits, dim=0)  # Shape: [sequence_length, vocab_size]
            batch_logits.append(sample_logits)  # Collect logits
            batch_targets.append(single_label)  # Collect labels

        # Concatenate logits and targets for the batch
        output_logits = torch.cat(batch_logits, dim=0)  # Shape: [batch_size * sequence_length, vocab_size]
        flattened_target = torch.cat(batch_targets, dim=0)  # Shape: [batch_size * sequence_length]

        print(f"Final output_logits shape: {output_logits.shape}")  # Should match target shape
        print(f"Final flattened_target shape: {flattened_target.shape}")

        # Calculate and backpropagate loss
        if output_logits.shape[0] == flattened_target.shape[0]:
            loss = criterion(output_logits, flattened_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch {batch_idx + 1} Loss: {loss.item()}")
        else:
            print("Shape mismatch between logits and targets; skipping batch.")
    
    # Log batch loss to W&B
    wandb.log({
        "batch_loss": loss.item(),
        "batch_idx": batch_idx + 1,
        "epoch": epoch + 1
    })
    
    avg_loss = total_loss / len(dataloader)
    
    wandb.log({
        "epoch_loss": avg_loss,
        "epoch": epoch + 1
    })
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")


wandb.finish()
