import shutil
import unittest
import os

from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir

experiment_config = {
    "output_path": "/tmp",
    "data_saver_config": {},  # provide empty dict for default data_saving config, if None --> no data saved.
    "number_of_episodes": 2,
    "architecture_config": {
        "architecture": "tiny_128_rgb_6c",
        "initialisation_type": 'xavier',
        "random_seed": 0,
        "device": 'cpu'},
    "tensorboard": False,
    "environment_config": {
      "factory_key": "ROS",
      "max_number_of_steps": 3,
      "ros_config": {
        "info": [
            "current_waypoint",
            "sensor/odometry"
        ],
        "observation": "sensor/forward_camera",
        "visible_xterm": False,
        "step_rate_fps": 30,
        "ros_launch_config": {
          "random_seed": 123,
          "robot_name": "turtlebot_sim",
          "fsm_config": "single_run",  # file with fsm params loaded from config/fsm
          "fsm": True,
          "control_mapping": True,
          "waypoint_indicator": True,
          "control_mapping_config": "evaluation",
          "world_name": "cube_world",
          "x_pos": 0.0,
          "y_pos": 0.0,
          "z_pos": 0.0,
          "yaw_or": 1.57,
          "gazebo": True,
        },
        "actor_configs": [{
              "name": "ros_expert",
              "file": "src/sim/ros/config/actor/ros_expert.yml"
            }],
      },
    },
}


class TestRosModelEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)

    def test_ros_with_model_evaluation(self):
        experiment_config['output_path'] = self.output_dir
        self.experiment = Experiment(ExperimentConfig().create(config_dict=experiment_config))
        self.experiment.run()
        raw_data_dirs = [os.path.join(self.output_dir, 'raw_data', d)
                         for d in os.listdir(os.path.join(self.output_dir, 'raw_data'))]
        self.assertEqual(len(raw_data_dirs), 1)
        run_dir = raw_data_dirs[0]
        with open(os.path.join(run_dir, 'done.data'), 'r') as f:
            self.assertEqual(experiment_config["number_of_episodes"] *
                             experiment_config["environment_config"]["max_number_of_steps"],
                             len(f.readlines()))
        self.experiment.shutdown()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
