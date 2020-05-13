#!/usr/python
import sys
import time

start_time = time.time()
duration = 60
while time.time() < start_time + duration:
    print(f'{time.time()} : sleeping')
    time.sleep(1)

sys.exit(0)
