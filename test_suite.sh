#!/usr/bin/env bash

echo "
Test all code base ROS exclusive from low to high dependent code.
"
start_time="$(date)"

source virtualenvironment/venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

if [ -d test_dir ] ; then
  rm -r test_dir;
fi
mkdir test_dir

EXCLUDE=" "
# EXCLUDE="src/scripts/test/test_model_evaluation_ros.py"



# src/core src/data src/ai src/sim/common src/sim/gym src/scripts src/condor
for group in src/core src/data src/ai src/sim/common src/sim/gym src/scripts src/condor ; do
  echo "$(date +%H:%M:%S) ------- ${group}"
  for test in "${group}"/test/test_*.py ; do
    [[ ${EXCLUDE} =~ (^|[[:space:]])${test}($|[[:space:]]) ]]
    if [ $? = 1 ] ; then
      echo "${test} "
      destination=test_dir/"$(basename "$test" | cut -d '.' -f 1)"
      if [ -d "$destination" ]; then
        rm -r "$destination"
      fi
      python "${test}" > "$destination".out 2>&1
      exitcode=$?
      if [ $exitcode = 0 ] ; then
        OK=$(cat "$destination".out | grep OK)
        echo " --> ${OK}"
      else
        echo "ERROR $exitcode --> $(tail ${destination}.out)"
      fi
    else
      echo "EXCLUDE ${test}"
    fi
  done
done
echo "FINISHED: $start_time -> $(date)"
