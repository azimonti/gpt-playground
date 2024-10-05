#!/usr/bin/env python3
'''
/********************/
/* mod_config.py    */
/*   Version 2.0    */
/*    2024/01/04    */
/********************/
'''
from types import SimpleNamespace

tokenizer_cfg = SimpleNamespace(
    # Use the basic tokenizer
    USE_BASIC_TOKENIZER=False
)

training_model_cfg = SimpleNamespace(
    # Batch sizes
    TRAIN_BATCH_SIZE=16,
    EVAL_BATCH_SIZE=8,
    # Number of tokens processed in a single sequence
    CONTEXT_LENGTH=512,
    # Percentage of data to use for training
    TRAIN_SPLIT=0.7,
    # Learning rate
    LEARNING_RATE=1e-5,
    # Used to define size of embeddings
    D_MODEL=1024,
    # Number of epochs
    NUM_EPOCHS=100,
    # Number of workers
    NUM_WORKERS=1,
    # Number of threads
    NUM_THREADS=1,
    # Validation Epoch Step
    EVAL_EPOCH_STEP=10,
    # Use the basic model
    USE_BASIC_MODEL=False,
    # Maximum length of input sequences
    MAX_LENGTH=512,
    # Hidden dimension
    HIDDEN_DIMENSION=2048,
    # Use multiple-head attention
    USE_MULTIPLE_HEAD=True,
    # Number of multiple heads
    NUM_HEADS=4,
    # Number of layers
    N_LAYERS=8,
    # Dropout probability
    DROPOUT_PROB=0.25
)

tokenizer_cfg = SimpleNamespace(
    # Use the basic tokenizer
    USE_BASIC_TOKENIZER=False
)

run_model_cfg = SimpleNamespace(
    # Use the basic model
    USE_BASIC_MODEL=False,
    # Number of new tokens to generate
    MAX_NEW_TOKENS=50,
    # Temperature for sampling (controls randomness)
    TEMPERATURE=1.0,
    # Top-k sampling (limits candidate tokens)
    TOP_K=10
)

introspect_model_cfg = SimpleNamespace(
    # Use the basic model
    USE_BASIC_MODEL=False
)

plot_cfg = SimpleNamespace(
    # Plot the validation loss
    PLOT_VALIDATION_LOSS=False,
    # Use Simple Moving Average (SMA) for smoothing the loss
    USE_SMA=False,
    # Alpha for exponential moving average (EMA)
    EMA_ALPHA=0.5,
    # Set the window size for SMA
    SMA_WINDOW_SIZE=10
)

if __name__ == "__main__":
    pass
