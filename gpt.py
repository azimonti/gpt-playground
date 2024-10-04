#!/usr/bin/env python3
'''
/****************/
/*   gpt.py     */
/*  Version 2.0 */
/*  2024/10/04  */
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


class MyGPTBlock(nn.Module):
    """
    Defines a single GPT block, which includes
            multi-head attention and feed-forward network.
    """

    def __init__(self, d_model, hidden_dim, dropout_prob, num_heads,
                 use_multiple_head):
        super().__init__()
        # Attention layer
        if use_multiple_head:
            self.attention = MultiHeadAttention(d_model, num_heads)
        else:
            self.attention = SingleHeadAttention(d_model)

        # Feedforward neural network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # Expansion
            nn.GELU(),                      # Activation
            nn.Linear(hidden_dim, d_model)   # Compression
        )
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # Dropout for regularization (optional)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Apply attention and layer norm
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)  # Residual connection

        # Apply feed-forward network and layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)   # Residual connection
        x = self.dropout(x)         # Apply dropout
        return x


class MyGPT(nn.Module):
    """
    Stacks multiple GPT blocks to create a full GPT model.
    """

    def __init__(self, vocab_size, d_model, max_len, hidden_dim,
                 dropout_prob, n_layers, num_heads,
                 use_multiple_head):
        """
        Initializes the modular GPT model with stacked layers.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - d_model (int): Dimensionality of embeddings and model.
        - num_heads (int): Number of attention heads for multi-head attention.
        - max_len (int): Maximum length of input sequences.
        - hidden_dim (int): Hidden layer dimension for feed-forward network.
        - n_layers (int): Number of stacked GPT blocks (transformer layers).
        - use_multiple_head (bool): Whether to use multi-head attention
                or single-head.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.use_multiple_head = use_multiple_head

        # Word token embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.pe = PositionalEncoding(d_model, max_len)

        # Stack multiple GPT blocks
        self.blocks = nn.ModuleList([
            MyGPTBlock(d_model, hidden_dim, dropout_prob,
                       num_heads, use_multiple_head)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        # Output projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, targets=None):
        """
        Forward pass through the model.

        Parameters:
        - inputs (torch.Tensor): Input tensor of token indices.
        - targets (torch.Tensor, optional): Target tensor of token
                indices for loss computation.

        Returns:
        - logits (torch.Tensor): The output predictions (logits) of the model.
        - loss (torch.Tensor or None): The computed loss if targets
                are provided, otherwise None.
        """
        # (batch_size, sequence_length, d_model)
        embeddings = self.wte(inputs)

        # Apply final layer normalization and output projection
        # (batch_size, sequence_length, vocab_size)
        logits = self.fc_out(self.ln_f(embeddings))

        loss = None
        if targets is not None:
            # Flatten logits and targets for loss computation
            batch_size, sequence_length, vocab_size_out = logits.shape
            logits = logits.reshape(
                batch_size * sequence_length, vocab_size_out)
            targets = targets.reshape(batch_size * sequence_length)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens, temperature,
                 top_k, context_length):
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
        - output (torch.Tensor): Generated sequence including
            the input sequence.
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
            # Forward pass through the model
            logits, _ = self(inputs)  # Only use the logits, ignore the loss
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

    def introspect(self, compile_model=True, device='cpu'):
        """
        Prints the model architecture, parameters, and the total
                number of parameters.
        Optionally compiles the model using `torch.compile`.

        Parameters:
        - compile_model (bool): If True, compiles the model
                using torch.compile().
        """
        # Move the model to the specified device
        self.to(device)
        # Compile the model if requested
        if compile_model:
            self = torch.compile(self)

        # Print the model architecture
        print(self)

        # Print model configuration
        print(f"\nModel Configuration:\n"
              f"Vocab Size: {self.vocab_size}\n"
              f"Embedding Dim (d_model): {self.d_model}\n"
              f"Max Sequence Length: {self.max_len}\n"
              f"Hidden Dimension: {self.hidden_dim}\n"
              f"Dropout Probability: {self.dropout_prob}\n"
              f"Number of Attention Heads: {self.num_heads}\n"
              f"Number of Layers: {self.n_layers}\n"
              f"Using Multi-Head Attention: {self.use_multiple_head}\n")

        # Calculate total trainable parameters
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        print(f"Total Parameters: {total_params:,} "
              f"({round(total_params / 1_000_000)}M)\n")


if __name__ == "__main__":
    pass
