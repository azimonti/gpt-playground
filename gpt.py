#!/usr/bin/env python3
'''
/****************/
/*   gpt.py     */
/*  Version 2.0 */
/*  2024/10/02  */
/****************/
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()  # Correct parent init call
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)
        return out


class FeedForwardNN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def top_k_logits(logits, k):
    # Only keep the top k logits,
    # set the rest to -inf so they don't get sampled
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class MyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000, hidden_dim=2048):
        super().__init__()
        # Word token embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.pe = PositionalEncoding(d_model, max_len)
        # Attention layer
        self.attention = SingleHeadAttention(d_model)
        # Feedforward neural network
        self.ffn = FeedForwardNN(d_model, hidden_dim)
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, targets=None):
        # (batch_size, sequence_length, d_model)
        embeddings = self.wte(inputs)
        # Add positional encoding
        embeddings = self.pe(embeddings)
        # Apply attention
        attn_output = self.attention(embeddings)
        # Pass through feedforward NN
        ffn_output = self.ffn(attn_output)
        # (batch_size, sequence_length, vocab_size)
        logits = self.fc_out(self.ln_f(ffn_output))

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size_out = logits.shape
            logits = logits.reshape(
                batch_size * sequence_length, vocab_size_out)
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
