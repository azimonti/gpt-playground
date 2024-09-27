#!/usr/bin/env python3
'''
/*********************/
/*    tokenizer.py   */
/*    Version 1.0    */
/*    2024/09/27     */
/*********************/
'''
import torch
# Read the input file
with open('./build/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the text by splitting on whitespace
tokens = text.split()

# Create a vocabulary mapping each unique token to a unique integer
vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}

# Convert the list of tokens into a list of token IDs
token_ids = [vocab[token] for token in tokens]

# Convert the list of token IDs into a PyTorch tensor
token_tensor = torch.tensor(token_ids, dtype=torch.long)

# Save the vocabulary and the token tensor using torch.save
torch.save({'vocab': vocab, 'token_tensor': token_tensor},
           './build/tokenized_data.pkl')

# Output some information about the tokenization
print("Vocabulary Size:", len(vocab))
print("First 10 Tokens:", tokens[:10])
print("First 10 Token IDs:", token_ids[:10])
print("Token Tensor Shape:", token_tensor.shape)
