#!/usr/bin/env python3
'''
/*********************/
/*    tokenizer.py   */
/*    Version 1.0    */
/*    2024/09/27     */
/*********************/
'''
import re
import unicodedata
import torch


def main():
    # Read the input file
    with open('./runs/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert the text to lowercase for consistency
    text = text.lower()

    # Custom preprocessing for numbers: Replace all numbers with a <NUM> token
    text = re.sub(r'\d+', '<NUM>', text)

    # Normalize the text to separate base characters and accents
    normalized_text = unicodedata.normalize('NFD', text)

    # Tokenize the text: match words, punctuation, and newlines
    tokens = re.findall(r'<[^>]+>|[\w]+|[^\w\s]|\n+', normalized_text)

    # Add special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    tokens = special_tokens + tokens

    # Create a vocabulary mapping each unique token to a unique integer
    vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}

    # Check if placeholder_string exists in the vocabulary
    # and replace it with <CANTO_END>
    if "placeholder_string" in tokens:
        tokens = ['<CANTO_END>' if token == 'placeholder_string'
                  else token for token in tokens]
        # Ensure "placeholder_string" is replaced in the vocabulary
        if "placeholder_string" in vocab:
            vocab["<CANTO_END>"] = vocab.pop("placeholder_string")

    # Convert the list of tokens into a list of token IDs
    token_ids = []
    for token in tokens:
        try:
            token_ids.append(vocab[token])
        except KeyError:
            print(f"Error: Token '{token}' not found in vocabulary.")

    # Convert the list of token IDs into a PyTorch tensor
    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Save the vocabulary and the token tensor using torch.save
    torch.save({'vocab': vocab, 'token_tensor': token_tensor},
               './runs/tokenized_data.pkl')

    # Output some information about the tokenization
    print("Vocabulary Size:", len(vocab))
    print("First 30 Tokens:", tokens[:30])
    print("First 30 Token IDs:", token_ids[:30])
    print("Token Tensor Shape:", token_tensor.shape)
    print("Vocabulary:", vocab)


if __name__ == "__main__":
    main()
