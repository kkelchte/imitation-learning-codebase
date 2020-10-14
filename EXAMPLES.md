# imitation-learning-codebase examples

## Train a DNN to drive the turtlebot through Cube-world 

Gazebo simulator is accessible within the singularity image. 
Launch a terminal or your pycharm environment in a singularity environment.

```bash
cd rosenvironment/singularity
make singularity-shell
source rosenvironment/entrypoint.sh
```

You can take a look at cube world in gazebo with the following command:
```bash
gazebo src/sim/ros/gazebo/worlds/cube_world.world
```
The robot spawns at the origin and the goal is to drive around the cube as specified in the following file:
`src/sim/ros/config/world/cube_world.yml`

_Collect data manually with keyboard_

Keyboard combination:
- i: go straight
- u: sharp turn left
- o: sharp turn right
- g: start recording
- f: reset and start a new run

The keybindings are defined in `src/sim/ros/config/actor/keyboard_turtlebot_sim.yml`.
After each run, so successfully reaching the goal state, 
the experiment goes in 'take over' mode which means it is not recording.
Resume recording by pressing 'g'. After three runs or recordings the software closes automatically.

```bash
python3.8 src/scripts/experiment.py --config src/scripts/config/il_data_collection_cube_world_interactive.yml
```

Take a look at the configuration file and feel free to change some parameters to discover the use.
After driving around, the data will be stored in $DATADIR/cube_world/raw_data.
In $DATADIR/cube_world/trajectories you have a top down view of the recorded trajectories.

You can automate the process of data collection in a simulation-supervised fashion by using extra sensors such as
the ground truth global position in combination with waypoints to navigate the robot according to a heuristic.
The heuristic is specified in `ros_expert.py`.
With the following command you add a couple of automated runs with the ros expert.
You can see that the expert is moving in a very shaky way, this is due to noise added by the `control_mapping.py`.
Due to the noise, the expert needs to recover from undesired mistakes and the data is much more informative.

```bash
python3.8 src/scripts/experiment.py --config src/scripts/config/il_data_collection_cube_world.yml
```

In order to view the console you'll have to set `robot_display: true` 
as was done in `il_data_collection_cube_world_interactive.yml`.
Alternatively you can visualise the robot in gazebo by calling up the client from a terminal:
```bash
gzclient
```

_Clean data and create hdf5 file_

This command collects all the data from the `cube_world/raw_data` directory and stores them in a hdf5 file. 
Before running this command, go through the runs and see if everything is stored correctly. 
If you have a run with very few data, just remove it and record some more.

```bash
python3.8 src/scripts/data_cleaning.py --config src/scripts/config/il_data_cleaning_cube_world.yml
```

This script can later be used to automatically clean or detect inconsistencies in your data. 
However, as that is task dependent it is not part of the master branch.

_Train a network and visualise results in Tensorboard_

Take a look at the config file `il_train_cube_world.yml` to see the hyperparameters and losses used for training.
The network architecture details are in `tiny_128_rgb_6c.py`.

```bash
python3.8 src/scripts/experiment.py --config src/scripts/config/il_train_cube_world.yml
```

The training graphs are kept in an Tensorboard events file which is interpreted with tensorboard:
For the moment tensorboard does not work in the singularity image, so run it in the virtual environment 
from a different terminal:

```bash
cdcodebase
# or if you don't have the alias set in your bash:
# source virtualenv/venv/bin/activate && export PYTHONPATH=$HOME/code/imitation-learning-codebase
tensorboard --logdir $DATADIR/cube_world
```

From you favorite browser, go to localhost:6006 to see the results.

_Evaluate you network trained in the gazebo environment_

```bash
python3.8 src/scripts/experiment.py --config src/scripts/config/il_evaluate_interactive_cube_world.yml
```