#!/usr/bin/env python3
'''
/********************/
/*  mod_logging.py  */
/*   Version 1.0    */
/*    2024/09/30    */
/********************/
'''

import time


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
        print(f"Message: {message}")


if __name__ == "__main__":
    pass
