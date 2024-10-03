#!/usr/bin/env python3
'''
/******************/
/*  run_model.py  */
/*   Version 1.0  */
/*    2024/09/29  */
/******************/
'''
import io
import torch
import sys
import argparse
from gpt_basic import MyGPT as GPT_basic
from gpt import MyGPT as GPT_v2
from mod_config import run_model_cfg as cfg
import tiktoken

# Fix encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def basic_generation(model, start_sequence, max_new_tokens, temperature,
                     top_k, vocab, id_to_token, device):
    """
    Basic generation using the current implementation.
    """
    # Encode the start sequence
    input_ids = [vocab.get(token, vocab["<UNK>"]) for token
                 in start_sequence.split()]
    input_ids = torch.tensor(
        input_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Generate new tokens using temperature and top_k
    generated_ids = model.generate(input_ids, max_new_tokens,
                                   temperature=temperature, top_k=top_k)

    # Decode the generated token IDs back to text
    return [id_to_token[idx] for idx in generated_ids.squeeze(0).tolist()]


def advanced_generation(model, start_sequence, max_new_tokens, vocab,
                        device, temperature=1.0, top_k=10):
    """
    Advanced generation using tiktoken,
    ensuring the token IDs match the model's vocabulary.
    """
    with torch.no_grad():
        # Initialize tiktoken's GPT-2 tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        # Encode the start sequence using tiktoken to get token IDs
        input_ids = tokenizer.encode(start_sequence)
        # Map the token IDs from tiktoken directly to your model's vocab
        model_input_ids = [vocab.get(tokenizer.decode([id]).strip(),
                                     vocab.get("<UNK>")) for id in input_ids]
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(
            model_input_ids, dtype=torch.long, device=device).unsqueeze(0)
        # Generate new tokens using the model's built-in generation method
        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k)
        # Reverse the vocab dictionary to map token IDs back to tokens
        id_to_token = {idx: token for token, idx in vocab.items()}
        # Convert generated token IDs back to tokens
        generated_tokens = [id_to_token.get(idx.item(), "<UNK>")
                            for idx in generated_ids.squeeze(0).tolist()]
        # Join tokens to form the generated text
        generated_text = ' '.join(generated_tokens)

        return generated_text


def main(use_simple_generation, max_new_tokens, temperature, top_k):
    # Load the tokenizer data (vocab and token tensor)
    tokenizer_data = torch.load('./runs/tokenized_data.pkl',
                                weights_only=False)
    vocab = tokenizer_data['vocab']
    # Reverse the vocab dictionary for decoding (used in basic_generation)
    id_to_token = {idx: token for token, idx in vocab.items()}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model checkpoint
    checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    use_simple_model = checkpoint['use_simple_model']

    # Initialize the model with loaded parameters
    if use_simple_model:
        model = GPT_basic(vocab_size=vocab_size, d_model=d_model).to(device)
    else:
        use_multiple_head = checkpoint.get('use_multiple_head', False)
        num_heads = checkpoint.get('num_heads', 8)
        max_length = checkpoint.get('max_length', 5000)
        hidden_dimension = checkpoint.get('hidden_dimension', 2048)
        model = GPT_v2(
            vocab_size=vocab_size, d_model=d_model, max_len=max_length,
            hidden_dim=hidden_dimension, use_multiple_head=use_multiple_head,
            num_heads=num_heads).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Get the start sequence
    start_sequence = "Nel mezzo del cammin di nostra vita"
    # Use basic or advanced generation
    if use_simple_generation:
        print("Using basic generation")
        generated_sequence = basic_generation(
            model, start_sequence, max_new_tokens, temperature,
            top_k, vocab, id_to_token, device=device)
        # Output the generated text
        print("Generated sequence:", " ".join(generated_sequence))
    else:
        print("Using tiktoken-based generation")
        generated_text = advanced_generation(
            model, start_sequence, max_new_tokens, vocab, device)
        # Output the generated text
        print("Generated sequence:", generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--use_simple_generation", action="store_true",
        default=cfg.USE_SIMPLE_GENERATION,
        help="Whether to use the simple generation")
    parser.add_argument(
        "-n", "--max_new_tokens", type=int, default=cfg.MAX_NEW_TOKENS,
        help="Number of new tokens to generate")
    parser.add_argument(
        "-t", "--temperature", type=float, default=cfg.TEMPERATURE,
        help="Temperature for sampling (controls randomness)")
    parser.add_argument(
        "-k", "--top_k", type=int, default=cfg.TOP_K,
        help="Top-k sampling (limits candidate tokens)")

    args = parser.parse_args()

    main(use_simple_generation=args.use_simple_generation,
         max_new_tokens=args.max_new_tokens,
         temperature=args.temperature,
         top_k=args.top_k)
