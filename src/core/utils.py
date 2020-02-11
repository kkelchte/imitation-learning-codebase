import os
from datetime import datetime


def camelcase_to_snake_format(input: str) -> str:
    output = ''
    prev_c = ''
    for c in input:
        if c.isupper():
            output += '_' + c.lower() if prev_c.isalpha() else c.lower()
        else:
            output += c
        prev_c = c
    return output


def get_filename_without_extension(filename: str) -> str:
    return str(os.path.basename(filename).split('.')[0])


def get_date_time_tag() -> str:
    return str(datetime.strftime(datetime.now(), format="%y-%m-%d_%H-%M-%S"))
