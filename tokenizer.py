#!/usr/bin/env python3
'''
/*********************/
/*    tokenizer.py   */
/*    Version 1.0    */
/*    2024/09/27     */
/*********************/
'''
import argparse
import io
import re
import torch
import sys
from mod_config import tokenizer_cfg as cfg
import tiktoken

# Fix encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def read_txt():
    # Read the input file
    with open('./runs/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def basic_tokenizer():
    text = read_txt()
    # Custom preprocessing for numbers: Replace all numbers with a <NUM> token
    text = re.sub(r'\d+', '<NUM>', text)
    # Tokenize the text: match words, punctuation, and newlines
    tokens = re.findall(r'<[^>]+>|[\w]+|[^\w\s]|\n+', text)
    # Add special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    tokens = special_tokens + tokens
    # Create a vocabulary mapping each unique token to a unique integer
    vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
    return tokens, vocab


def advanced_tokenizer():
    text = read_txt()
    # Custom preprocessing for numbers: Replace all numbers with a <NUM> token
    text = re.sub(r'\d+', '<NUM>', text)
    # Initialize tiktoken's GPT-3 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    # Add special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    # Tokenize the text and get the token IDs
    token_ids = enc.encode(text)
    # Decode back to tokens for clarity
    tokens = enc.decode(token_ids).split()
    # Prepend the special tokens to the list of tokens
    tokens = special_tokens + tokens
    # Create a vocabulary mapping each unique token to a unique integer
    vocab = {token: idx for idx, token in enumerate(sorted(set(tokens)))}
    return tokens, vocab


def main(use_basic_tokenizer):
    print("Using basic tokenizer" if use_basic_tokenizer else
          "Using tiktoken")

    if use_basic_tokenizer:
        tokens, vocab = basic_tokenizer()
    else:
        tokens, vocab = advanced_tokenizer()

    # Convert the list of tokens into a list of token IDs
    token_ids = [vocab.get(token, vocab.get("<UNK>")) for token in tokens]

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
    # print("Vocabulary", vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--use_basic_tokenizer", action="store_true",
        default=cfg.USE_BASIC_TOKENIZER,
        help="Whether to use the basic tokenizer")

    args = parser.parse_args()

    main(args.use_basic_tokenizer)
