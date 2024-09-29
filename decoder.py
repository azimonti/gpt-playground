#!/usr/bin/env python3
'''
/******************/
/*   decoder.py   */
/*   Version 1.0  */
/*    2024/09/29  */
/******************/
'''

import torch
from gpt import MyGPT as GPT

# Load the tokenizer data (vocab and token tensor)
tokenizer_data = torch.load('./build/tokenized_data.pkl', weights_only=False)
vocab = tokenizer_data['vocab']

# Reverse the vocab dictionary to decode token IDs back to text
id_to_token = {idx: token for token, idx in vocab.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model configuration and state_dict
checkpoint = torch.load('./build/gpt_model.pth', weights_only=False)
vocab_size = checkpoint['vocab_size']
d_model = checkpoint['d_model']

# Initialize the model with loaded parameters
model = GPT(vocab_size=vocab_size, d_model=d_model).to(device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Generate text function
def generate_text(model, start_sequence, max_new_tokens):
    # Encode the start sequence
    input_ids = [vocab[token] for token in start_sequence.split()]
    input_ids = torch.tensor(
        input_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = model.generate(input_ids, max_new_tokens)
    # Decode to text
    return [id_to_token[idx] for idx in generated_ids.squeeze(0).tolist()]


# Example usage
start_sequence = "Nel mezzo del cammin di nostra vita"
generated_sequence = generate_text(model, start_sequence, max_new_tokens=50)

# Split between the start sequence and the generated text
generated_only = generated_sequence[len(start_sequence.split()):]

print("Start sequence:", start_sequence)
print("Generated sequence:", " ".join(generated_only))

