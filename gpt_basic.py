#!/usr/bin/env python3
'''
/****************/
/* gpt_basic.py */
/*  Version 1.0 */
/*  2024/09/27  */
/****************/
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def top_k_logits(logits, k):
    # Only keep the top k logits,
    # set the rest to -inf so they don't get sampled
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class MyGPT(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Word token embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, targets=None):
        # (batch_size, sequence_length, d_model)
        embeddings = self.wte(inputs)
        # (batch_size, sequence_length, vocab_size)
        logits = self.fc_out(self.ln_f(embeddings))

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size_out = logits.shape
            logits = logits.reshape(batch_size * sequence_length,
                                    vocab_size_out)
            targets = targets.reshape(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=10):
        # Ensure generated_ids is a flat list
        generated_ids = input_ids.squeeze(0).tolist()
        context_length = input_ids.size(1)

        for _ in range(max_new_tokens):
            # Prepare the input tensor for the model
            input_tensor = torch.tensor(
                generated_ids[-context_length:],
                dtype=torch.long).unsqueeze(0).to(input_ids.device)
            # Forward pass through the model
            logits, _ = self.forward(input_tensor)
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            # Apply top-k sampling to limit the candidates
            next_token_logits = top_k_logits(next_token_logits, top_k)
            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            # Sample the next token from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            # Append the predicted token ID to the sequence
            generated_ids.append(next_token_id)

        return torch.tensor(
            generated_ids,
            dtype=torch.long).unsqueeze(0).to(input_ids.device)


if __name__ == "__main__":
    pass
