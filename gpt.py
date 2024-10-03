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
        seq_length = x.size(1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Apply causal mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length),
                          diagonal=1).bool().to(x.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        # Softmax normalization for attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(attention_weights, V)
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        # Dimensionality of each attention head
        self.head_dim = d_model // num_heads
        # Linear layers to project input into query, key, and value vectors
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # Final linear layer to project concatenated heads back to d_model
        self.fc_out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, seq_length, d_model = x.shape
        # Project input into query, key, and value vectors
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Split Q, K, V into multiple heads
        # Shape: (B, num_heads, seq_length, head_dim)
        Q = Q.view(B, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(B, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(B, seq_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Apply causal mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length),
                          diagonal=1).bool().to(x.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        # Softmax normalization for attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(attention_weights, V)
        # Concatenate the attention heads
        attention_output = attention_output.transpose(
            1, 2).contiguous().view(B, seq_length, d_model)
        # Final linear layer to project concatenated heads back to d_model
        out = self.fc_out(attention_output)

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
    # Only keep the top k logits, set the rest to -inf so they don't get
    # sampled
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class MyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000,
                 hidden_dim=2048, use_multiple_head=True, num_heads=8):
        """
        Initializes the GPT model

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - d_model (int): The dimensionality of the embeddings and model.
        - num_heads (int): The number of attention heads
                for multi-head attention (default: 8)
        - max_len (int): The maximum length of input sequences (default: 5000)
        - hidden_dim (int): The dimensionality of the hidden layer
                in the feedforward network (default: 2048).
        - use_multiple_head (bool): If True, uses multi-head attention;
                if False, uses single-head attention (default: True).
        """
        super().__init__()
        # Word token embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.pe = PositionalEncoding(d_model, max_len)

        # Attention layer: Switch between single-head or multi-head attention
        if use_multiple_head:
            self.attention = MultiHeadAttention(d_model, num_heads)
        else:
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
        # Apply attention (either single or multi-head based on init flag)
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

    def generate(self, inputs, max_new_tokens, temperature=1.0,
                 top_k=None, context_length=None):
        """
        Generate new tokens based on the input sequence

        Parameters:
        - inputs (torch.Tensor): The input sequence as token IDs.
        - max_new_tokens (int): Number of new tokens to generate.
        - temperature (float): Controls the randomness of predictions
                by scaling logits.
        - top_k (int): If specified, applies top-k sampling
                to limit the candidate tokens.
        - context_length (int): Length of the context window
                for the input sequence.

        Returns:
        - output (torch.Tensor): Generated sequence
                including the input sequence.
        """
        # Initialize the output with the input sequence
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = output.size(1)
            # Truncate inputs to respect the context length if specified
            if context_length and current_seq_length > context_length:
                inputs = output[:, -context_length:]
            else:
                inputs = output

            # Forward pass through the model (doesn't need to pass targets)
            logits, _ = self(inputs)
            # Get logits for the last token in the sequence
            logits = logits[:, -1, :]
            # Apply temperature scaling to control randomness
            logits = logits / temperature
            # Optionally apply top-k sampling to limit candidate tokens
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # Append the predicted token to the sequence
            output = torch.cat([output, next_token], dim=1)
        return output


if __name__ == "__main__":
    pass
