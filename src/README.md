# Source code for imitation-learning code base

The subdirectories have the following structure:

- _core_ : contains basic helper functions, has no other code-base dependencies except within core.
- _sim_ : contains code relevant for each simulated environment:
    - _gazebo_ : interfaces and helper functions for ROS-gazebo
        - _messages_ : define topics and message types for sensors and actions
        - _actors_ : defines potential actors interfacing with the environment.
        - _environment-generators_ : define world generators for building an environment.
        - _environments_ :
            - _object-models_
            - _textures_
    - (later) _gym_ : interfaces and helper functions for OpenAI gym.
    - (later) _carla_ : interfaces and helper functions for Carla.
    - (later) _sim4cv_ : interfaces and helper functions for sim4cv.
- _ai_ : contains the deep learning related code.
    - _components_ : separate deep learning components without dependencies within ai: 
        - _metrics_
        - (later) _losses_
        - (later) _noise_
    - _utils_ : helper functions to download, visualize, ... .
    - _data_ : code related to loading and preprocessing stored data.
    - _architectures_ : 
        - (later) _encoders_ : various (pretrained) encoding architectures.
        - (later) _decoders_ : various auxiliary outputs.
        - _models_ : integrated neural networks combining encoders and decoders.
    - _train_ : code related to training a model.
    - _evaluate_ : code related to evaluating a model.
    - _scripts_ : high-level data-collection, train and evaluation scripts.
- _condor_ : contains scripts and potentially services for launching and monitoring condor jobs.
    - _scripts_
    - _services_