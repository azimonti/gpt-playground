#!/usr/bin/env python3
'''
/******************/
/*  run_model.py  */
/*   Version 2.0  */
/*    2024/10/04  */
/******************/
'''
import io
import torch
import sys
import argparse
from gpt_basic import MyGPT as GPT_basic
from gpt import MyGPT as GPT_v2
from mod_config import run_model_cfg as cfg
from tokenizer import advanced_tokenizer

# Fix encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def basic_generation(model, start_sequence, max_new_tokens, device,
                     temperature, top_k):
    """
    Basic generation using basic tokenization (splitting by spaces).
    """
    # Load the tokenizer data (vocab and token tensor)
    tokenizer_data = torch.load('./runs/tokenized_data.pkl',
                                weights_only=False)
    vocab = tokenizer_data['vocab']
    # Reverse the vocab dictionary for decoding (used in basic_generation)
    id_to_token = {idx: token for token, idx in vocab.items()}
    # Use basic tokenization (splitting by spaces)
    input_ids = [vocab.get(token, vocab["<UNK>"])
                 for token in start_sequence.split()]
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(
        input_ids, dtype=torch.long, device=device).unsqueeze(0)
    # Generate new tokens using the model's generate function
    generated_ids = model.generate(
        input_tensor, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k)
    # Reverse the vocab dictionary to map token IDs back to tokens
    generated_tokens = [id_to_token.get(idx, "<UNK>")
                        for idx in generated_ids.squeeze(0).tolist()]
    # Join tokens to form the generated text
    generated_text = ' '.join(generated_tokens)
    return generated_text


def advanced_generation(model, start_sequence, max_new_tokens, device,
                        temperature, top_k, context_length):
    """
    Advanced generation using tiktoken, ensuring the token IDs
    match the model's vocabulary and applying temperature and top_k sampling.
    """
    with torch.no_grad():
        # Tokenize the text using the advanced tokenizer
        tokens, vocab = advanced_tokenizer()
        # Convert start_sequence tokens into token IDs using the custom vocab
        model_input_ids = [vocab.get(token, vocab.get("<UNK>"))
                           for token in start_sequence.split()]
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(
            model_input_ids, dtype=torch.long, device=device).unsqueeze(0)
        # Generate new tokens using your model
        generated_ids = model.generate(
            input_tensor, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            context_length=context_length)
        # Reverse the vocab dictionary to map token IDs back to tokens
        id_to_token = {idx: token for token, idx in vocab.items()}
        # Convert generated token IDs back to tokens
        generated_tokens = [id_to_token.get(idx.item(), "<UNK>")
                            for idx in generated_ids.squeeze(0)]
        # Only append generated tokens, not the start sequence
        generated_text = ' '.join(generated_tokens)
        return generated_text


def main(use_basic_model, max_new_tokens, temperature, top_k):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model checkpoint
    checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']

    # Initialize the model with loaded parameters
    if use_basic_model:
        model = GPT_basic(vocab_size=vocab_size, d_model=d_model).to(device)
    else:
        use_multiple_head = checkpoint.get('use_multiple_head')
        num_heads = checkpoint.get('num_heads')
        max_length = checkpoint.get('max_length')
        hidden_dimension = checkpoint.get('hidden_dimension')
        n_layers = checkpoint.get('n_layers')
        dropout_prob = checkpoint.get('dropout_prob')
        context_length = checkpoint.get('context_length')
        model = GPT_v2(
            vocab_size=vocab_size, d_model=d_model, max_len=max_length,
            hidden_dim=hidden_dimension, dropout_prob=dropout_prob,
            n_layers=n_layers, num_heads=num_heads,
            use_multiple_head=use_multiple_head).to(device)

    # Load the model state
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Get the start sequence
    start_sequence = "Nel mezzo del cammin di nostra vita"

    if use_basic_model:
        print("Using basic generation")
        generated_text = basic_generation(
            model, start_sequence, max_new_tokens, device, temperature, top_k)
    else:
        print("Using tiktoken-based generation")
        # Call the advanced_generation function (with temperature and top_k
        # handling)
        generated_text = advanced_generation(
            model, start_sequence, max_new_tokens, device, temperature, top_k,
        context_length)
    # Output the generated text
    print("Generated sequence:", generated_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--use_basic_model", action="store_true",
        default=cfg.USE_BASIC_MODEL,
        help="Whether to use the basic model")
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

    main(use_basic_model=args.use_basic_model,
         max_new_tokens=args.max_new_tokens,
         temperature=args.temperature,
         top_k=args.top_k)
