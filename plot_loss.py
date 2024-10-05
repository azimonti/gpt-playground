#!/usr/bin/env python3
'''
/******************/
/*  plot_loss.py  */
/*   Version 2.0  */
/*    2024/10/04  */
/******************/
'''
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pickle
from mod_config import plot_cfg as cfg


def simple_moving_average(data, window_size):
    sma = []
    for i in range(len(data)):
        if i < window_size:
            # Average up to the current point if the window isn't full
            sma.append(sum(data[:i + 1]) / (i + 1))
        else:
            sma.append(sum(data[i - window_size + 1:i + 1]) / window_size)
    return sma


def exponential_moving_average(data, alpha):
    ema = []
    ema.append(data[0])  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return ema


def create_axes(ax, y_data):
    ax.axis("on")
    ax.grid(linewidth=0.4, linestyle="--", dashes=(5, 20))
    max_value = max(y_data)
    if max_value > 1:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    else:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{x:.2e}'))


def main(logdir, plot_val_loss, use_sma, ema_alpha, sma_window_size):
    output_file = os.path.join(logdir, 'output.pkl')
    # Check for file existence and raise an error if it's missing
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"{output_file} not found.")
    with open(output_file, 'rb') as f:
        data = pickle.load(f)
    x_train, y_train = zip(*data['Training Loss'])
    x_val, y_val = zip(*data['Validation Loss'])
    # Compute smoothed version
    if use_sma:
        y_train_smoothed = simple_moving_average(y_train, sma_window_size)
        y_val_smoothed = simple_moving_average(y_val, sma_window_size)
    else:
        y_train_smoothed = exponential_moving_average(y_train, ema_alpha)
        y_val_smoothed = exponential_moving_average(y_val, ema_alpha)
    # Plotting
    ck = (44 / 255, 44 / 255, 44 / 255)
    if plot_val_loss:
        fig = plt.figure(figsize=(8, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        create_axes(ax1, y_train)
        c = (102 / 255, 204 / 255, 255 / 255)
        ax1.plot(x_train, y_train_smoothed, label="Training Loss Smoothed",
                 color=c, linewidth=1.0)
        ax1.plot(x_train, y_train, label="Training Loss", color=ck,
                 linewidth=3.0)
        ax1.plot(x_train, y_train, label="Training Loss", color=c,
                 linewidth=2.0)
        ax1.set_title("Training Loss")

        ax2 = fig.add_subplot(1, 2, 2)
        create_axes(ax2, y_val)
        c = (255 / 255, 153 / 255, 102 / 255)
        ax2.plot(x_val, y_val_smoothed, label="Validation Loss Smoothed",
                 color=c, linewidth=1.0)
        ax2.plot(x_val, y_val, label="Validation Loss", color=ck,
                 linewidth=3.0)
        ax2.plot(x_val, y_val, label="Validation Loss", color=c,
                 linewidth=2.0)
        ax2.set_title("Validation Loss")
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot()
        create_axes(ax1, y_train)
        create_axes(ax1, y_train)
        ax1.plot(x_train, y_train_smoothed, label="Training Loss Smoothed",
                 color=c, linewidth=1.0)
        c = (102 / 255, 204 / 255, 255 / 255)
        ax1.plot(x_train, y_train, label="Training Loss", color=ck,
                 linewidth=3.0)
        ax1.plot(x_train, y_train, label="Training Loss", color=c,
                 linewidth=2.0)
        ax1.set_title("Training Loss")

    plt.tight_layout()
    plt.savefig(logdir + '/loss.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--logdir", type=str, required=True,
        help='Path to the log directory')
    parser.add_argument(
        "-vl", "--plot_val_loss", action='store_true',
        default=cfg.PLOT_VALIDATION_LOSS, help='Plot Validation Loss')
    parser.add_argument(
        "-s", "--use_sma", action='store_true',
        default=cfg.USE_SMA,
        help='Use Simple Moving Average for smoothing the loss')
    parser.add_argument(
        "-a", "--ema_alpha", type=float,
        default=cfg.EMA_ALPHA,
        help='Set the alpha value for Exponential Moving Average (EMA)')
    parser.add_argument(
        "-w", "--sma_window_size", type=int,
        default=cfg.SMA_WINDOW_SIZE,
        help='Set the window size for Simple Moving Average')

    args = parser.parse_args()

    main(
        logdir=args.logdir,
        plot_val_loss=args.plot_val_loss,
        use_sma=args.use_sma,
        ema_alpha=args.ema_alpha,
        sma_window_size=args.sma_window_size)
