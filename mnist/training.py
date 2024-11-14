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

# Prepare dataset and dataloaders
dataset = CombineDataset()

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Update your dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
encoder = Encoder(dim_size, num_heads, num_layers)
decoder = Decoder(vocab_size, dim_size, num_heads, num_layers)

# Move models to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)

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
    total_accuracy = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
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

        # Get predicted labels
        predicted_labels = output.argmax(dim=-1)  # [batch_size, sequence_length]

        # Compute loss
        loss = criterion(output.view(-1, vocab_size), labels.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Compute accuracy
        correct_predictions = (predicted_labels == labels).float()
        batch_accuracy = correct_predictions.sum() / correct_predictions.numel()

        total_loss += loss.item()
        total_accuracy += batch_accuracy.item()

        # Log batch loss and accuracy
        wandb.log({
            "batch_loss": loss.item(),
            "batch_accuracy": batch_accuracy.item(),
            "epoch": epoch + 1
        })

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
              f"Loss: {loss.item():.4f}, Accuracy: {batch_accuracy.item():.4f}")

        # Optionally, log images and predictions to W&B
        if batch_idx % 10 == 0:  # Adjust the frequency as needed
            for i in range(min(4, batch_size)):
                wandb.log({
                    "examples": [
                        wandb.Image(images[i].cpu(), caption=f"Predicted: {predicted_labels[i].tolist()}, "
                                                             f"Actual: {labels[i].tolist()}")
                    ]
                })

    avg_loss = total_loss / min(len(train_loader), max_batches)
    avg_accuracy = total_accuracy / min(len(train_loader), max_batches)
    wandb.log({
        "epoch_loss": avg_loss,
        "epoch_accuracy": avg_accuracy,
        "epoch": epoch + 1
    })
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}, "
          f"Average Accuracy: {avg_accuracy:.4f}")

    # Validation loop
    encoder.eval()
    decoder.eval()
    val_loss = 0
    val_accuracy = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            encoder_output = encoder(images)
            decoder_input = torch.zeros(images.size(0), sequence_length, dtype=torch.long, device=device)
            output = decoder(decoder_input, encoder_output)

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
            val_loss += loss.item()

            # Compute accuracy
            predicted_labels = output.argmax(dim=-1)
            correct_predictions = (predicted_labels == labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            val_accuracy += accuracy.item()

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)

    # Log validation metrics
    wandb.log({
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy,
        "epoch": epoch + 1
    })

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

wandb.finish()
