#!/usr/python3.7
import shlex
import subprocess
import sys

from src.core.utils import *

subprocess.call(shlex.split("printenv"))
print('well done!')
sys.exit(2)
