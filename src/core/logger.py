import logging
import os

from datetime import datetime
from enum import IntEnum

from src.core.utils import get_date_time_tag
"""Modified logger to store easy-extractable log files.

"""


def get_logger(name: str, output_path: str = '', quite: bool = False) -> logging.Logger:
    logger = logging.getLogger(name=name)
    [logger.removeHandler(handler) for handler in logger.handlers]

    # add file handler
    if output_path != '':
        os.makedirs(os.path.join(output_path, 'log_files'), exist_ok=True)
        output_file = os.path.join(output_path, 'log_files', f'{get_date_time_tag()}_{name}.log')
    else:
        output_file = f'/tmp/{datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M")}.log'
    f_handler = logging.FileHandler(output_file)
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    # add stream handler
    if not quite:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    logger.setLevel(logging.DEBUG)
    return logger


class MessageType(IntEnum):
    info = 0
    error = 1
    warning = 2
    debug = 3


def cprint(message: str,
           logger: logging.Logger = None,
           msg_type: MessageType = MessageType.info) -> None:
    if logger is None:
        print(f'{msg_type.name}: {message}')
    else:
        eval(f'logger.{msg_type.name}(message)')
