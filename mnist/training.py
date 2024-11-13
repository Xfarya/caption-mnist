import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn

from mnist.encoderdecoder import Decoder, Encoder

vocab_size = 10
dim_size = 64
seq_length = 5
batch_size = 1

encoder = Encoder(vocab_size, dim_size)
decoder = Decoder(vocab_size, dim_size)


input_sequence = torch.tensor([[1, 2, 3, 4]])
target_sequence = torch.tensor([[2, 3, 4, 5, 6]])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

encoded_output = encoder(input_sequence)

decoder_input = torch.tensor([[0]])

encoder.train()
decoder.train()
optimizer.zero_grad()

output_logits = decoder(decoder_input, encoded_output)

output_logits = output_logits.view(-1, vocab_size)
target_sequence = target_sequence.view(-1)

loss = criterion(output_logits, target_sequence)

loss.backward()
optimizer.step()

print("Loss:", loss.item())
