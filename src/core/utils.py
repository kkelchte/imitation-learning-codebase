#!/usr/python3.7

import os
import subprocess
from datetime import datetime


def get_file_length(file_path: str) -> int:
    with open(file_path, 'r') as f:
        return len(f.readlines())


def read_file_to_output(file_path: str) -> None:
    print('#' * 50 + ' ' * 5 + os.path.basename(file_path) + ' ' * 5 + '#' * 50)
    with open(file_path, 'r') as f:
        for line in f.readlines():
            print(line.strip())
    print('#' * 50 + ' ' * 5 + 'END' + ' ' * 5 + '#' * 50)


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


def count_grep_name(grep_str: str) -> int:
    ps_process = subprocess.Popen(["ps", "-ef"],
                                  stdout=subprocess.PIPE)
    with ps_process.stdout:
        grep_process = subprocess.Popen(["grep", grep_str],
                                        stdin=ps_process.stdout,
                                        stdout=subprocess.PIPE)
        with grep_process.stdout:
            output_string = str(grep_process.communicate()[0])
    processed_output_string = [line for line in output_string.split('\\n') if 'grep' not in line
                               and 'test' not in line and len(line) > len(grep_str) and 'pycharm' not in line]
    return len(processed_output_string)


def get_to_root_dir():
    # assume you're in a subfolder in the codebase:
    while 'ROOTDIR' not in os.listdir('.'):
        os.chdir('..')
        if os.getcwd() == '/':
            raise FileNotFoundError
