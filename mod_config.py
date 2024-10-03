#!/usr/bin/env python3
'''
/********************/
/* mod_config.py    */
/*   Version 1.1    */
/*    2024/01/02    */
/********************/
'''
from types import SimpleNamespace

model_training_cfg = SimpleNamespace(
    # Batch sizes
    TRAIN_BATCH_SIZE=16,
    EVAL_BATCH_SIZE=8,
    # Number of tokens processed in a single sequence
    CONTEXT_LENGTH=1024,
    # Percentage of data to use for training
    TRAIN_SPLIT=0.7,
    # Learning rate
    LEARNING_RATE=1e-3,
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
    # Use the simple model
    USE_BASIC_MODEL=False
)

tokenizer_cfg = SimpleNamespace(
    # Use the simple tokenizer
    USE_BASIC_TOKENIZER=False
)

if __name__ == "__main__":
    pass
