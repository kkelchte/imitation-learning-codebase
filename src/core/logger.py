import logging
import os

from datetime import datetime

"""Modified logger to store easy-extractable log files.

"""


def get_logger(name: str,
               output_path: str = '',
               quite: bool = False) -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'/tmp/{datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M")}.log' if not output_path else \
        f'{output_path}/logfile'
    f_handler = logging.FileHandler(output_file)

    c_handler.setLevel(logging.INFO if not quite else logging.ERROR)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    [logger.removeHandler(handler) for handler in logger.handlers]
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.DEBUG)

    return logger


def cprint(message: str, logger: logging.Logger = None) -> None:
    if logger is None:
        print(message)
    else:
        logger.info(message)
