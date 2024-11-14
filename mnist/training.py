import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

from encoderdecoder import Encoder, Decoder  # Ensure these are correctly implemented

# Initialize W&B project
wandb.init(project="caption-mnist", name="transformer-mnist")

# Model parameters
vocab_size = 10
dim_size = 64
batch_size = 8
num_epochs = 30
sequence_length = 4
num_heads = 8
num_layers = 3
learning_rate = 0.0001  # Decreased learning rate
max_batches = None  # Remove limit on batches

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

# Load the MNIST dataset once
full_mnist_dataset = torchvision.datasets.MNIST(
    root='.', download=True, transform=torchvision.transforms.ToTensor()
)

# Split MNIST dataset into training, validation, and test sets
mnist_train_size = int(0.7 * len(full_mnist_dataset))
mnist_val_size = int(0.15 * len(full_mnist_dataset))
mnist_test_size = len(full_mnist_dataset) - mnist_train_size - mnist_val_size

mnist_train_dataset, mnist_val_dataset, mnist_test_dataset = torch.utils.data.random_split(
    full_mnist_dataset, [mnist_train_size, mnist_val_size, mnist_test_size]
)

# Define the CombineDataset class with data leakage prevention
class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset, num_samples):
        super().__init__()
        self.dataset = mnist_dataset
        self.length = num_samples  # Number of combinations to generate

        # Pre-generate combinations to ensure consistency
        self.indices_list = []
        for _ in range(self.length):
            indices = random.sample(range(len(self.dataset)), 4)
            self.indices_list.append(indices)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        indices = self.indices_list[idx]
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

# Create CombineDataset instances for training, validation, and test sets
train_num_samples = 10000  # Adjust as needed
val_num_samples = 2000     # Adjust as needed
test_num_samples = 2000    # Adjust as needed

train_dataset = CombineDataset(mnist_train_dataset, train_num_samples)
val_dataset = CombineDataset(mnist_val_dataset, val_num_samples)
test_dataset = CombineDataset(mnist_test_dataset, test_num_samples)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# Initialize variables for early stopping and best validation accuracy
best_val_accuracy = 0.0
patience = 5
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Move data to device
        images = images.to(device)  # [batch_size, 1, 56, 56]
        labels = labels.to(device)  # [batch_size, sequence_length]

        # Forward pass through encoder
        encoder_output = encoder(images)  # [batch_size, encoder_seq_len, dim_size]

        # Prepare decoder input with teacher forcing
        decoder_input = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
        decoder_input[:, 1:] = labels[:, :-1]  # Shifted ground truth labels
        # The first input token is assumed to be 0 (<START> token)

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
        batch_accuracy_percent = batch_accuracy.item() * 100
        wandb.log({
            "batch_loss": loss.item(),
            "batch_accuracy": batch_accuracy_percent,
            "epoch": epoch + 1
        })

        # Incorporate visualizations into the training loop
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {batch_accuracy_percent:.2f}%")

            # Visualize and log examples to W&B
            num_examples = min(4, batch_size)
            images_cpu = images[:num_examples].cpu()
            predicted_labels_cpu = predicted_labels[:num_examples].cpu().tolist()
            labels_cpu = labels[:num_examples].cpu().tolist()

            for i in range(num_examples):
                img = images_cpu[i].squeeze()
                pred_labels = predicted_labels_cpu[i]
                true_labels = labels_cpu[i]
                caption = f"Predicted: {pred_labels}, Actual: {true_labels}"

                # Log image and caption to W&B
                wandb.log({
                    "Training Examples": wandb.Image(img, caption=caption)
                })

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_accuracy_percent = avg_accuracy * 100

    wandb.log({
        "epoch_loss": avg_loss,
        "epoch_accuracy": avg_accuracy_percent,
        "epoch": epoch + 1
    })
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}, "
          f"Average Accuracy: {avg_accuracy_percent:.2f}%")

    # Validation loop
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            encoder_output = encoder(images)
            decoder_input = torch.zeros(images.size(0), sequence_length, dtype=torch.long, device=device)
            decoder_input[:, 1:] = labels[:, :-1]  # Teacher forcing
            output = decoder(decoder_input, encoder_output)

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), labels.view(-1))
            val_loss += loss.item()

            # Compute accuracy
            predicted_labels = output.argmax(dim=-1)
            correct_predictions = (predicted_labels == labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            val_accuracy += accuracy.item()

            # Visualize and log validation examples
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(val_loader):
                num_examples = min(4, images.size(0))
                images_cpu = images[:num_examples].cpu()
                predicted_labels_cpu = predicted_labels[:num_examples].cpu().tolist()
                labels_cpu = labels[:num_examples].cpu().tolist()

                for i in range(num_examples):
                    img = images_cpu[i].squeeze()
                    pred_labels = predicted_labels_cpu[i]
                    true_labels = labels_cpu[i]
                    caption = f"Predicted: {pred_labels}, Actual: {true_labels}"

                    # Log image and caption to W&B
                    wandb.log({
                        "Validation Examples": wandb.Image(img, caption=caption)
                    })

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_accuracy / len(val_loader)
    avg_val_accuracy_percent = avg_val_accuracy * 100

    wandb.log({
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_accuracy_percent,
        "epoch": epoch + 1
    })

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy_percent:.2f}%")

    # Check for improvement
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        patience_counter = 0

        # Save the models
        torch.save(encoder.state_dict(), 'best_encoder.pth')
        torch.save(decoder.state_dict(), 'best_decoder.pth')

        print(f"New best validation accuracy: {avg_val_accuracy_percent:.2f}% - models saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Testing loop with visualizations
encoder.eval()
decoder.eval()
test_loss = 0.0
test_accuracy = 0.0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        encoder_output = encoder(images)
        decoder_input = torch.zeros(images.size(0), sequence_length, dtype=torch.long, device=device)
        decoder_input[:, 1:] = labels[:, :-1]  # Teacher forcing
        output = decoder(decoder_input, encoder_output)

        # Compute loss
        loss = criterion(output.view(-1, vocab_size), labels.view(-1))
        test_loss += loss.item()

        # Compute accuracy
        predicted_labels = output.argmax(dim=-1)
        correct_predictions = (predicted_labels == labels).float()
        accuracy = correct_predictions.sum() / correct_predictions.numel()
        test_accuracy += accuracy.item()

        # Visualize and log test examples
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(test_loader):
            num_examples = min(4, images.size(0))
            images_cpu = images[:num_examples].cpu()
            predicted_labels_cpu = predicted_labels[:num_examples].cpu().tolist()
            labels_cpu = labels[:num_examples].cpu().tolist()

            for i in range(num_examples):
                img = images_cpu[i].squeeze()
                pred_labels = predicted_labels_cpu[i]
                true_labels = labels_cpu[i]
                caption = f"Predicted: {pred_labels}, Actual: {true_labels}"

                # Log image and caption to W&B
                wandb.log({
                    "Test Examples": wandb.Image(img, caption=caption)
                })

# Calculate average test loss and accuracy
avg_test_loss = test_loss / len(test_loader)
avg_test_accuracy = test_accuracy / len(test_loader)
avg_test_accuracy_percent = avg_test_accuracy * 100

wandb.log({
    "test_loss": avg_test_loss,
    "test_accuracy": avg_test_accuracy_percent,
    "epoch": num_epochs  # Or another identifier
})

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy_percent:.2f}%")

# Finish the W&B run after all logging is done
wandb.finish()
