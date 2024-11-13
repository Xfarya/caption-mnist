import torch.optim as optim
import torch
import torch.nn as nn

from encoderdecoder import Decoder, Encoder

import wandb

# Model parameters
vocab_size = 10
dim_size = 64
seq_length = 5
batch_size = 1
num_epochs = 10

# Initialize W&B project
wandb.init(project="caption-mnist", name="1. initial")  # Customize the project and experiment names

wandb.config = {
    "epochs": num_epochs,
    "batch_size": batch_size,
    "vocab_size": vocab_size,
    "dim_size": dim_size,
    "learning_rate": 0.001,
}


# Initialize Encoder and Decoder
encoder = Encoder(vocab_size, dim_size)
decoder = Decoder(vocab_size, dim_size)

# Define a sample input and target sequence
input_sequence = torch.tensor([[1, 2, 3, 4, 5]])  # Shape: [1, 5]
target_sequence = torch.tensor([[2, 3, 4, 5, 6]])  # Shape: [1, 5]

# Print shapes after definition
print("Initial shape of input_sequence:", input_sequence.shape)      # Expected: [1, 5]
print("Initial shape of target_sequence:", target_sequence.shape)    # Expected: [1, 5]

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    # Encode the input sequence
    encoded_output = encoder(input_sequence)

    # Initialize the decoder with the <START> token
    decoder_input = torch.tensor([[0]], device=input_sequence.device)

    # Store output logits
    output_logits = []

    # Print shape of target_sequence before accessing its length
    print(f"Epoch {epoch+1}, target_sequence shape before loop:", target_sequence.shape)

    # Sequential decoding loop for each token position
    for i in range(target_sequence.size(1)):  # Using target sequence length dynamically
        logits = decoder(decoder_input, encoded_output)
        output_logits.append(logits)

        # Choose the highest probability token
        _, next_token = logits.max(dim=-1)
        decoder_input = next_token  # Use generated token as the next input

    # Concatenate logits and reshape for loss calculation
    output_logits = torch.cat(output_logits, dim=1)  # Shape: [batch_size, sequence_length, vocab_size]
    output_logits = output_logits.view(-1, vocab_size)  # Flatten to [batch_size * sequence_length, vocab_size]

    # Create a separate variable for flattened target_sequence
    flattened_target = target_sequence.view(-1)  # Flatten to [batch_size * sequence_length]
    print("Shape of flattened_target before loss:", flattened_target.shape)  # Expected: [5]

    # Calculate loss
    loss = criterion(output_logits, flattened_target)
    loss.backward()
    optimizer.step()

    wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

wandb.finish()

# Testing the model after training
print("\nGenerating sequence after training...\n")
encoder.eval()
decoder.eval()

with torch.no_grad():
    encoded_output = encoder(input_sequence)
    decoder_input = torch.tensor([[0]], device=input_sequence.device)  # <START> token
    generated_sequence = []

    for _ in range(seq_length):
        logits = decoder(decoder_input, encoded_output)
        _, next_token = logits.max(dim=-1)
        generated_sequence.append(next_token.item())
        decoder_input = next_token

print("Input sequence:", input_sequence.tolist())
print("Target sequence:", target_sequence.view(-1).tolist())
print("Generated sequence:", generated_sequence)
