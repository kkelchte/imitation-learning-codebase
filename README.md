# imitation-learning-codebase overview
integrated code base for imitation learning projects

Installation with ROS environment is defined in /rosenvironment by a singularity definition file installing all dependencies. For only using the pytorch AI code, installing in the virtualenvironment defined in /virtualenvironment should be enough. 
The Makefile are better used as references rather than blindly using it with make as sourcing or activitating environments are not conveniently done in a make command.

### /rosenvironment 
Directory contains the definition file of the singularity container as well as the build instructions in the make file.

### /virtualenvironment
Directory contains python virtual environment which requires to be sourced to run python scripts.

### /src
Directory contains source code of total project.

# Example usage

__imitation learning in cube world__

python3.8 src/scripts/experiment.py --config src/scripts/config/il_data_collection_cube_world.yml

python3.8 src/scripts/data_cleaning.py --config src/scripts/config/il_data_cleaning_cube_world.yml 

python3.8 src/scripts/experiment.py --config src/scripts/config/il_train_cube_world.yml 

python3.8 src/scripts/experiment.py --config src/scripts/config/il_evaluate_interactive_cube_world.yml 

# Troubleshoot

__Module Not Found__
1. make sure each python scripts is called from the main (this) directory
2. make sure the correct python3.8 interpreter is used