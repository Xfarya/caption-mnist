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

        return img, label

# Model parameters
vocab_size = 10
dim_size = 64
batch_size = 1  # Use 1 batch size to get one combined image of 4 digits
num_epochs = 500

# Initialize W&B project
wandb.init(project="caption-mnist", name="single-image-test")

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

# Initialize the Combine dataset and DataLoader with batch_size=1
ds = Combine()
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop for 1 combined image (one batch) only
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0

    # Only process a single batch and then break
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx > 0:
            break  # Stop after one batch
        
        optimizer.zero_grad()

        # Encode the input single combined image
        encoded_output = encoder(images)

        # Initialize the decoder with the <START> token
        decoder_input = torch.zeros((batch_size, 1), dtype=torch.long)

        # Initialize empty list for output logits
        output_logits = []

        # Loop through each label index based on the actual length of `labels[0]`
        for i in range(len(labels[0])):  # Use the number of labels in the first (and only) list in the batch
            logits = decoder(decoder_input, encoded_output)  # Decoder output for the current token
            output_logits.append(logits.squeeze(1))  # Remove extra dimension to ensure shape [batch_size, vocab_size]

            # Update decoder_input with the next token in the sequence
            decoder_input = torch.tensor([labels[0][i]], device=logits.device).unsqueeze(1)

        # Stack logits along the sequence length dimension
        output_logits = torch.stack(output_logits, dim=0).view(-1, vocab_size)  # Expected shape: [4, vocab_size]

        # Flatten target labels to match output_logits
        flattened_target = torch.tensor(labels[0], device=logits.device).view(-1)  # Expected shape: [4]

        # Debug print statements for shape verification
        print(f"output_logits shape: {output_logits.shape}")  # Expected: [4, vocab_size]
        print(f"flattened_target shape: {flattened_target.shape}")  # Expected: [4]

        # Calculate loss
        loss = criterion(output_logits, flattened_target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

wandb.finish()
