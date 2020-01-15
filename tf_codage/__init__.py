import os

def print_env():
    """Dump environment to the standard output"""
    for k, v in os.environ.items():
        print('{}={}'.format(k, v))