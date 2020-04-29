import shutil
import unittest
import os


from src.scripts.experiment import ExperimentConfig, Experiment
from src.core.utils import get_filename_without_extension, get_to_root_dir

model_evaluation_config = {
    "output_path": "/tmp",
    "data_saver_config": {},  # provide empty dict for default data_saving config, if None --> no data saved.
    "number_of_episodes": 2,
    "model_config": {
        "architecture": "tiny_net_v1",
        "input_sizes": [3, 128, 128],
        "output_sizes": [1]
    },
    "tensorboard": False,
    "environment_config": {
      "factory_key": 0,
      "max_number_of_steps": 10,
      "ros_config": {
        "info": [
            "current_waypoint",
            "sensor/odometry"
        ],
        "observation": "sensor/forward_camera",
        "visible_xterm": True,
        "step_rate_fps": 2,
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

    def test_experiment_with_empty_config(self):
        self.config = ExperimentConfig().create(
            config_dict={
                'output_path': self.output_dir
            }
        )
        self.experiment = Experiment(self.config)
        self.experiment.run()
        self.experiment.shutdown()

    def test_ros_with_model_evaluation(self):
        model_evaluation_config['output_path'] = self.output_dir
        self.config = ExperimentConfig().create(
            config_dict=model_evaluation_config
        )
        self.experiment = Experiment(self.config)
        self.experiment.run()
        self.experiment.shutdown()

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
