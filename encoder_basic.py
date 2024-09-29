#!/usr/bin/env python3
'''
/********************/
/* encoder_basic.py */
/*   Version 1.0    */
/*    2024/09/29    */
/********************/
'''
from gpt import MyGPT as GPT
import multiprocessing as mp
import sys
import time
import torch
from torch.utils.data import DataLoader, Dataset


# Parameters
train_batch_size = 16
eval_batch_size = 8
# Number of tokens processed in a single sequence
context_length = 1024
train_split = 0.7  # Percentage of data to use for training
learning_rate = 1e-3
# used to define size of embeddings
d_model = 1024
# Number of epochs
num_epochs = 100
# Number of workers
nw = 6


def print_time(t1, message):
    # Calculate elapsed time
    t2 = time.time()
    elapsed_time = t2 - t1
    minutes, seconds = divmod(int(elapsed_time), 60)
    # Print the message with the time in mm:ss format
    print(f"{message}: {minutes:02}:{seconds:02}")
    # Return the current time (t2) for further tracking
    return t2


class TokenDataset(Dataset):
    # DataSet Class
    def __init__(self, data_tensor, context_length):
        self.data_tensor = data_tensor
        self.context_length = context_length

    def __len__(self):
        # Number of sequences of `context_length` we can extract
        return len(self.data_tensor) // self.context_length

    def __getitem__(self, idx):
        # Return a chunk of `context_length` tokens
        start = idx * self.context_length
        end = start + self.context_length
        return self.data_tensor[start:end]


def main():
    start_time = time.time()
    t1 = print_time(start_time, "Start..")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load('./build/tokenized_data.pkl', weights_only=False)
    token_tensor = data['token_tensor']
    vocab = data['vocab']
    vocab_size = len(vocab)

    # Split the data for training and evaluation
    train_size = int(len(token_tensor) * train_split)
    train_tensor = token_tensor[:train_size]
    eval_tensor = token_tensor[train_size:]

    # DataLoader
    train_dataset = TokenDataset(train_tensor, context_length)
    eval_dataset = TokenDataset(eval_tensor, context_length)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=nw)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                             shuffle=False, num_workers=nw)

    t1 = print_time(t1, "Start training..")
    # Initialize model
    model = GPT(vocab_size=vocab_size, d_model=d_model).to(device)
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # Take all tokens except the last one as inputs
            inputs = batch[:, :-1].to(device)
            # Shift inputs by one token for targets
            targets = batch[:, 1:].to(device)

            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_tl = total_loss / len(train_loader)
        t1 = print_time(
            t1, f"Epoch [{epoch}/{num_epochs}], Loss: {avg_tl:.4f}")

        # Evaluate model on validation data
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)
                logits, loss = model(inputs, targets)
                eval_loss += loss.item()

        avg_evl = eval_loss / len(eval_loader)
        t1 = print_time(
            t1, f"Epoch [{epoch}/{num_epochs}], Eval Loss: {avg_evl:.4f}")

    # Save model configuration and state_dict together
    torch.save({
        'vocab_size': vocab_size,
        'd_model': d_model,
        'state_dict': model.state_dict()
    }, './build/gpt_model.pth')

    print_time(start_time, "End.")


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.set_start_method('spawn')
        torch.set_num_threads(6)
    main()
