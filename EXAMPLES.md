# imitation-learning-codebase examples

## Create a new simulated environment and collect data with a flying expert

This tutorial requires access to the gazebo simulator and ros which is accessible from within the singularity image.

### Create a world and save it

A Gazebo world is defined in a world file. See `src/sim/ros/gazebo/worlds` for examples.
The world file defines objects, a ground plane and the user camera used by the client.
You can create a new world model from within the gazebo interface with `gazebo` command.
On the left, second tab, you can insert models from the model database. 
And you can just save the environment in the world model.
Gazebo comes with a model builder to draw walls and doors and windows. 

You can also create a new model in any 3D editor engine such as blender.
The engine creates a mesh which can be loaded in the model file. 
See `src/sim/ros/gazebo/models` for example model files.
These models can then be loaded in the world file.
You can launch gazebo environment with `gazebo worldfile.world`.

### Define expert behavior in this environment

The ros flying expert is defined in the following file:
 `src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/ros_expert.py`.
It combines a waypoint behavior with a collision avoidance behavior.
The waypoints of the environment are defined in the world config file, such as:
`src/sim/ros/config/world/cube_world.yml`. 
The config file specifies also the goal area. 
If the agent reaches this area the rosnode, the episode ends.
The expert avoids obstacles by reading out a 360degree laser range finder and turning away from the closest obstacles.

In order to test the behavior of the expert in your newly created environment you can run 
the following tests in `src/sim/ros/test`:
`validate_expert_in_environment.py`, `navigate_robots_with_keyboard.py`.

### Collecting data over episodes run by the expert

Copy and adjust config `src/scripts/config/il_data_collection_cube_world.yml` and run experiment:
`python src/script/experiments.py --config src/script/config/YOURCONFIG.yml`
The data is stored in the outputpath subdirectory raw_data.
You can create a hdf5_file with the script:
 `src/scripts/data_cleaning.py --config src/scripts/config/YOURDATACLEANINGCONFIG.yml` 



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

### Collect data manually with keyboard

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

### Clean data and create hdf5 file

This command collects all the data from the `cube_world/raw_data` directory and stores them in a hdf5 file. 
Before running this command, go through the runs and see if everything is stored correctly. 
If you have a run with very few data, just remove it and record some more.

```bash
python3.8 src/scripts/data_cleaning.py --config src/scripts/config/il_data_cleaning_cube_world.yml
```

This script can later be used to automatically clean or detect inconsistencies in your data. 
However, as that is task dependent it is not part of the master branch.

### Train a network and visualise results in Tensorboard

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

### Evaluate you network trained in the gazebo environment

```bash
python3.8 src/scripts/experiment.py --config src/scripts/config/il_evaluate_interactive_cube_world.yml
```