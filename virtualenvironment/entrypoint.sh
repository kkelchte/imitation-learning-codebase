#!/usr/bin/env bash

CODEDIR="${PWD}"
PYTHONPATH=${PWD}:${PWD}/../dummy-tracking-gym
"$@"