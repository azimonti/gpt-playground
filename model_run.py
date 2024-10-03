#!/usr/bin/env python3
'''
/******************/
/*  model_run.py  */
/*   Version 1.0  */
/*    2024/09/29  */
/******************/
'''
import io
import torch
import sys
from gpt_basic import MyGPT as GPT_basic
from gpt import MyGPT as GPT_v2

# Fix encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')


def main():
    # Load the tokenizer data (vocab and token tensor)
    tokenizer_data = torch.load('./runs/tokenized_data.pkl',
                                weights_only=False)
    vocab = tokenizer_data['vocab']

    # Reverse the vocab dictionary to decode token IDs back to text
    id_to_token = {idx: token for token, idx in vocab.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    use_simple_model = checkpoint['use_simple_model']

    # Initialize the model with loaded parameters
    if (use_simple_model):
        model = GPT_basic(vocab_size=vocab_size, d_model=d_model).to(device)
    else:
        model = GPT_v2(vocab_size=vocab_size, d_model=d_model).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    def generate_text(model, start_sequence, max_new_tokens,
                      temperature=1.0, top_k=10):
        # Encode the start sequence
        input_ids = [vocab.get(token, vocab["<UNK>"]) for token in
                     start_sequence.split()]
        input_ids = torch.tensor(
            input_ids, dtype=torch.long).unsqueeze(0).to(device)
        # Generate new tokens using temperature and top_k
        generated_ids = model.generate(
            input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
        # Decode the generated token IDs back to text
        return [id_to_token[idx] for idx in generated_ids.squeeze(0).tolist()]

    # Example usage
    start_sequence = "Nel mezzo del cammin di nostra vita"
    generated_sequence = generate_text(model, start_sequence,
                                       max_new_tokens=50)

    # Split between the start sequence and the generated text
    generated_only = generated_sequence[len(start_sequence.split()):]

    print("Start sequence:", start_sequence)
    print("Generated sequence:", " ".join(generated_only))


if __name__ == "__main__":
    main()
