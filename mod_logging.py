#!/usr/bin/env python3
'''
/********************/
/*  mod_logging.py  */
/*   Version 1.0    */
/*    2024/09/30    */
/********************/
'''
import atexit
import pickle
import os
import time
from torch.utils.tensorboard import SummaryWriter


class UtilityLogger:
    _instance = None
    _shared_state = {}

    def __new__(cls):
        # Singleton
        if cls._instance is None:
            cls._instance = super(UtilityLogger, cls).__new__(cls)
            cls._instance.__dict__ = cls._shared_state
        return cls._instance

    @classmethod
    def set_variable(cls, key, value):
        cls._shared_state[key] = value

    @classmethod
    def get_variable(cls, key):
        return cls._shared_state.get(key, None)

    @classmethod
    def print_time(cls, t1, message):
        t2 = time.time()
        elapsed_time = t2 - t1
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            print(f"{message}: {hours:02}:{minutes:02}:{seconds:02}")
        else:
            print(f"{message}: {minutes:02}:{seconds:02}")
        return t2

    @classmethod
    def print_message(cls, message):
        print(f"{message}")


class PickleLogger:
    _instance = None
    _shared_state = {}
    _pickle_buffer = {}
    _last_flush_time = time.time()

    def __new__(cls, log_dir=None, flush_secs=120):
        if cls._instance is None:
            cls._instance = super(PickleLogger, cls).__new__(cls)
            cls._instance.__dict__ = cls._shared_state
            if log_dir is None:
                raise ValueError("log_dir must be provided for PickleLogger")
            cls._instance.log_dir = log_dir
            cls._instance.flush_secs = flush_secs
            # Ensure close is called at program exit
            atexit.register(cls.close)
        return cls._instance

    @classmethod
    def add_scalar(cls, tag, value, step):
        # Add scalar to pickle buffer
        if tag not in cls._pickle_buffer:
            cls._pickle_buffer[tag] = []
        cls._pickle_buffer[tag].append((step, value))

        # Check if we need to flush the buffer to disk
        current_time = time.time()
        if current_time - cls._last_flush_time >= cls._instance.flush_secs:
            cls._flush_pickle_data()

    @classmethod
    def _flush_pickle_data(cls):
        # Ensure log_dir is valid
        if cls._instance.log_dir is None:
            raise ValueError("log_dir is None, cannot flush pickle data")

        # Define the path for the pickle file
        pickle_file_path = os.path.join(cls._instance.log_dir, 'output.pkl')

        # Load existing data from disk if available
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}

        # Merge buffer with existing data
        for tag, values in cls._pickle_buffer.items():
            if tag not in data:
                data[tag] = []
            data[tag].extend(values)

        # Write the combined data to disk
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(data, f)

        # Clear the buffer and update the last flush time
        cls._pickle_buffer.clear()
        cls._last_flush_time = time.time()

    @classmethod
    def close(cls):
        # Flush remaining data before exit
        cls._flush_pickle_data()


class TorchLogger:
    _instance = None
    _shared_state = {}
    _closed = False

    def __new__(cls, log_dir=None, flush_secs=120):
        if cls._instance is None:
            cls._instance = super(TorchLogger, cls).__new__(cls)
            cls._instance.__dict__ = cls._shared_state

            # Initialize SummaryWriter
            cls._instance.writer = SummaryWriter(log_dir=log_dir)
            cls._instance.log_dir = cls._instance.writer.log_dir
            # Initialize PickleLogger with SummaryWriter log_dir
            cls._instance.pickle_logger = PickleLogger(
                log_dir=cls._instance.log_dir, flush_secs=flush_secs)
            atexit.register(cls.close)
        return cls._instance

    @classmethod
    def get_log_dir(cls):
        return cls._instance.log_dir

    @classmethod
    def add_scalar(cls, tag, value, step):
        # Log to TensorBoard (SummaryWriter)
        cls._instance.writer.add_scalar(tag, value, step)
        # Log to PickleLogger
        cls._instance.pickle_logger.add_scalar(tag, value, step)

    @classmethod
    def close(cls):
        if not cls._closed:
            cls._instance.writer.close()
            cls._instance.pickle_logger.close()
            cls._closed = True


if __name__ == "__main__":
    pass
