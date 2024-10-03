#!/usr/bin/env python3
'''
/*********************/
/* training_model.py */
/*    Version 1.1    */
/*     2024/10/02    */
/*********************/
'''
import argparse
from gpt_basic import MyGPT as GPT_basic
from gpt import MyGPT as GPT_v2
import multiprocessing as mp
import time
import torch
from torch.utils.data import DataLoader, Dataset
from mod_config import training_model_cfg as cfg
from mod_logging import TorchLogger, UtilityLogger as ul


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
         learning_rate, d_model, num_epochs, nw, nt, eval_epoch_step,
         use_simple_model, max_length, hidden_dimension, use_multiple_head,
         num_heads, continue_training):
    start_time = time.time()
    if use_simple_model:
        print("Using basic model")
    else:
        if use_multiple_head:
            print("Using advanced model with multi-head attention")
        else:
            print("Using advanced model with single-head attention")

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
    if (use_simple_model):
        model = GPT_basic(vocab_size=vocab_size, d_model=d_model).to(device)
    else:
        model = GPT_v2(
            vocab_size=vocab_size, d_model=d_model, max_len=max_length,
            hidden_dim=hidden_dimension, use_multiple_head=use_multiple_head,
            num_heads=num_heads).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    log_dir = None
    # If continue_training is set to True, load the model and the parameters
    if continue_training:
        checkpoint = torch.load('./runs/gpt_model.pth', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['num_epochs']  # Load the epoch number
        vocab_size = checkpoint.get('vocab_size')  # Load vocab size
        d_model = checkpoint.get('d_model')  # Load model dimension
        log_dir = checkpoint.get('log_dir', None)  # Load log directory
        use_multiple_head = checkpoint.get('use_multiple_head', False)
        num_heads = checkpoint.get('num_heads', 8)  # Load number of heads
        max_length = checkpoint.get('max_length', 5000)  # Load max length
        hidden_dimension = checkpoint.get('hidden_dimension', 2048)

    # Initialize TorchLogger
    logger = TorchLogger(log_dir=log_dir)

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
                logger.add_scalar(
                    'Training Loss',
                    loss.item(), epoch * len(train_loader) + i)

        avg_tl = total_loss / len(train_loader)
        t1 = ul.print_time(
            t1, f"Epoch [{epoch}/{start_epoch + num_epochs}], "
            f"Loss: {avg_tl:.4f}")

        if epoch % eval_epoch_step == 0:
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
            logger.add_scalar('Validation Loss', avg_evl, epoch)
            t1 = ul.print_time(
                t1, f"Epoch [{epoch}/{start_epoch + num_epochs}], "
                f"Eval Loss: {avg_evl:.4f}")

        # Calculate and display the estimated completion time
        elapsed_time = t1 - ts
        # Epochs completed in the current run
        completed_epochs = epoch + 1 - start_epoch
        remaining_epochs = (start_epoch + num_epochs) - (epoch + 1)
        # Remaining epochs in the current run
        est_completion = time.time() + (
            elapsed_time / completed_epochs * remaining_epochs)
        completion_time = time.strftime(
            "%H:%M", time.localtime(est_completion))
        ul.print_message(f"Estimated completion at {completion_time}")

    # Save model and the parameters
    torch.save({
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_epochs': num_epochs + start_epoch,
        'state_dict': model.state_dict(),
        'log_dir': TorchLogger.get_log_dir(),
        'use_simple_model': use_simple_model,
        'use_multiple_head': use_multiple_head,  # Save multiple head setting
        'num_heads': num_heads,  # Save number of heads
        'max_length': max_length,  # Save max length
        'hidden_dimension': hidden_dimension  # Save hidden dimension
    }, './runs/gpt_model.pth')

    # Close the logger when done
    TorchLogger.close()
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
        "-es", "--eval-epoch-step", type=int, default=cfg.EVAL_EPOCH_STEP,
        help="Number of threads")
    parser.add_argument(
        "-s", "--use_simple_model", action="store_true",
        default=cfg.USE_BASIC_MODEL,
        help="Whether to use the simple model")
    parser.add_argument(
        "-c", "--continue_training", action="store_true", default=False,
        help="Whether to continue training from a checkpoint")
    parser.add_argument(
        "-ml", "--max-length", type=int, default=cfg.MAX_LENGTH,
        help="Maximum length of input sequences")
    parser.add_argument(
        "-hd", "--hidden-dimension", type=int, default=cfg.HIDDEN_DIMENSION,
        help="Hidden dimension size")
    parser.add_argument(
        "-umh", "--use-multiple-head", action="store_true",
        default=cfg.USE_MULTIPLE_HEAD,
        help="Whether to use multiple-head attention")
    parser.add_argument(
        "-nh", "--num-heads", type=int, default=cfg.NUM_HEADS,
        help="Number of multiple heads")
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
        eval_epoch_step=args.eval_epoch_step,
        use_simple_model=args.use_simple_model,
        max_length=args.max_length,
        hidden_dimension=args.hidden_dimension,
        use_multiple_head=args.use_multiple_head,
        num_heads=args.num_heads,
        continue_training=args.continue_training
    )
