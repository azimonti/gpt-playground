#!/usr/bin/env python3
'''
/********************/
/* mod_config.py    */
/*   Version 1.1    */
/*    2024/01/02    */
/********************/
'''
from types import SimpleNamespace

tokenizer_cfg = SimpleNamespace(
    # Use the simple tokenizer
    USE_BASIC_TOKENIZER=False
)

training_model_cfg = SimpleNamespace(
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
    USE_BASIC_MODEL=False,
    # Maximum length of input sequences
    MAX_LENGTH=5000,
    # Hidden dimension
    HIDDEN_DIMENSION=2048,
    # Use multiple-head attention
    USE_MULTIPLE_HEAD=True,
    # Number of multiple heads
    NUM_HEADS=8
)


tokenizer_cfg = SimpleNamespace(
    # Use the simple tokenizer
    USE_BASIC_TOKENIZER=False
)

run_model_cfg = SimpleNamespace(
    # Whether to use the simple generation
    USE_SIMPLE_GENERATION=False,
    # Number of new tokens to generate
    MAX_NEW_TOKENS=50,
    # Temperature for sampling (controls randomness)
    TEMPERATURE=1.0,
    # Top-k sampling (limits candidate tokens)
    TOP_K=10
)

if __name__ == "__main__":
    pass
