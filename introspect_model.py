#!/usr/bin/env python3
'''
/***********************/
/* introspect_model.py */
/*     Version 2.0     */
/*      2024/10/04     */
/***********************/
'''
import io
import torch
import sys
import argparse
from gpt_basic import MyGPT as GPT_basic
from gpt import MyGPT as GPT_v2
from mod_config import introspect_model_cfg as cfg

# Fix encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def main(use_basic_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model checkpoint
    checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    # Initialize the model with loaded parameters
    if use_basic_model:
        print("Using basic model")
        model = GPT_basic(vocab_size=vocab_size, d_model=d_model).to(device)
    else:
        use_multiple_head = checkpoint.get('use_multiple_head')
        if use_multiple_head:
            print("Using advanced model with multi-head attention")
        else:
            print("Using advanced model with single-head attention")
        num_heads = checkpoint.get('num_heads')
        max_length = checkpoint.get('max_length')
        hidden_dimension = checkpoint.get('hidden_dimension')
        n_layers = checkpoint.get('n_layers')
        dropout_prob = checkpoint.get('dropout_prob')
        model = GPT_v2(
            vocab_size=vocab_size, d_model=d_model, max_len=max_length,
            hidden_dim=hidden_dimension, dropout_prob=dropout_prob,
            n_layers=n_layers, num_heads=num_heads,
            use_multiple_head=use_multiple_head).to(device)
    model.introspect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--use_basic_model", action="store_true",
        default=cfg.USE_BASIC_MODEL,
        help="Whether to use the basic model")
    args = parser.parse_args()
    main(use_basic_model=args.use_basic_model)
