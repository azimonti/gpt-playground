#!/usr/bin/env python3
'''
/********************/
/* encoder_basic.py */
/*   Version 1.0    */
/*    2024/09/29    */
/********************/
'''
import argparse
from gpt import MyGPT as GPT
import multiprocessing as mp
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from mod_config import basic_cfg as cfg
from mod_logging import UtilityLogger as ul

# Initialize TensorBoard writer
writer = SummaryWriter()


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


def main(train_batch_size, eval_batch_size, context_length, train_split,
         learning_rate, d_model, num_epochs, nw, nt, continue_training):
    start_time = time.time()
    ul.set_variable('start_time', start_time)

    t1 = ul.print_time(start_time, "Continuing.." if continue_training
                       else "Start..")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load('./runs/tokenized_data.pkl', weights_only=False)
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
    if nw > 1:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=True, num_workers=nw)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                 shuffle=False, num_workers=nw)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                  shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                 shuffle=False)

    t1 = ul.print_time(start_time, "Continuing training.." if continue_training
                       else "Start training..")

    # Initialize model
    model = GPT(vocab_size=vocab_size, d_model=d_model).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0

    # If continue_training is set to True, load the model and optimizer states
    if continue_training:
        checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['num_epochs']  # Load the epoch number

    # Continue or start training
    ts = time.time()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            # Take all tokens except the last one as inputs
            inputs = batch[:, :-1].to(device)
            # Shift inputs by one token for targets
            targets = batch[:, 1:].to(device)

            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # Log training loss every N batches
            if i % 10 == 0:
                writer.add_scalar(
                    'Training Loss',
                    loss.item(), epoch * len(train_loader) + i)

        avg_tl = total_loss / len(train_loader)
        t1 = ul.print_time(
            t1, f"Epoch [{epoch}/{start_epoch + num_epochs}], "
            f"Loss: {avg_tl:.4f}")

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
        writer.add_scalar('Validation Loss', avg_evl, epoch)
        t1 = ul.print_time(
            t1, f"Epoch [{epoch}/{start_epoch + num_epochs}], "
            f"Eval Loss: {avg_evl:.4f}")
        # Calculate and display the estimated completion time
        elapsed_time = t1 - ts
        remaining_epochs = (start_epoch + num_epochs) - (epoch + 1)
        est_completion = time.time() + (
            elapsed_time / (epoch + 1) * remaining_epochs)
        completion_time = time.strftime(
            "%H:%M", time.localtime(est_completion))
        ul.print_message(f"Estimated completion at {completion_time}")

    torch.save({
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_epochs': num_epochs + start_epoch,
        'state_dict': model.state_dict()
    }, './runs/gpt_model.pth')
    writer.close()
    ul.print_time(start_time, "End.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-tbs", "--train_batch_size", type=int, default=cfg.TRAIN_BATCH_SIZE,
        help="Training batch size")
    parser.add_argument(
        "-ebs", "--eval_batch_size", type=int, default=cfg.EVAL_BATCH_SIZE,
        help="Evaluation batch size")
    parser.add_argument(
        "-cl", "--context_length", type=int, default=cfg.CONTEXT_LENGTH,
        help="Context length")
    parser.add_argument(
        "-ts", "--train_split", type=float, default=cfg.TRAIN_SPLIT,
        help="Train/test split percentage")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=cfg.LEARNING_RATE,
        help="Learning rate")
    parser.add_argument(
        "-dm", "--d_model", type=int, default=cfg.D_MODEL,
        help="Size of embeddings (d_model)")
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=cfg.NUM_EPOCHS,
        help="Number of epochs")
    parser.add_argument(
        "-nw", "--nw", type=int, default=cfg.NUM_WORKERS,
        help="Number of workers")
    parser.add_argument(
        "-nt", "--nt", type=int, default=cfg.NUM_THREADS,
        help="Number of threads")
    parser.add_argument(
        "-c", "--continue_training", action="store_true", default=False,
        help="Whether to continue training from a checkpoint")

    args = parser.parse_args()

    if args.nw > 1:
        mp.set_start_method('spawn')
    if args.nt > 1:
        torch.set_num_threads(args.nt)

    main(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        context_length=args.context_length,
        train_split=args.train_split,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_epochs=args.num_epochs,
        nw=args.nw,
        nt=args.nt,
        continue_training=args.continue_training
    )
