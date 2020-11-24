#!/usr/bin/env bash

export CODEDIR="${PWD}"
export PYTHONPATH=${PWD}:${PWD}/../dummy-tracking-gym
"$@"