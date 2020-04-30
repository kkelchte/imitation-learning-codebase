#!/usr/bin/env bash

echo "
Test all code base ROS exclusive from low to high dependent code.
"
date

source virtualenvironment/venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -d test_output ] ; then
  rm -r test_output;
fi
mkdir test_output

EXCLUDE="src/scripts/test/test_model_evaluation_ros.py"

for group in src/core src/data src/ai src/sim/common src/sim/gym src/scripts src/condor ; do
  echo "$(date +%H:%M:%S) ------- ${group}"
  for test in "${group}"/test/test_*.py ; do
    [[ ${EXCLUDE} =~ (^|[[:space:]])${test}($|[[:space:]]) ]]
    if [ $? = 1 ] ; then
      echo "${test} "
      python3.7 "${test}" > test_output/"$(basename "$test" | cut -d '.' -f 1)" 2>&1
      exitcode=$?
      if [ $exitcode = 0 ] ; then
        OK=$(cat "test_output/$(basename "$test" | cut -d '.' -f 1)" | grep OK)
        echo "exit 0 --> ${OK}"
      else
        echo "ERROR $exitcode --> $(tail "test_output/$(basename "$test" | cut -d '.' -f 1)")"
      fi
    else
      echo "EXCLUDE ${test}"
    fi
  done
done
echo "FINISHED"
date
