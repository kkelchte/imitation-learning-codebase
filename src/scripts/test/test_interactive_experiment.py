import shutil
import unittest
import os

import yaml

from src.scripts.interactive_experiment import InteractiveExperimentConfig, InteractiveExperiment
from src.core.utils import get_filename_without_extension


class TestRosExperiments(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/scripts/test/config/test_data_collection_in_ros_config.yml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        with open(os.path.join(self.output_dir, 'config.yml'), 'w') as f:
            yaml.dump(config_dict, f)
        self.config = InteractiveExperimentConfig().create(
            config_file=os.path.join(self.output_dir, 'config.yml')
        )

    def test_ros_with_data_collection(self):
        self.experiment = InteractiveExperiment(self.config)
        self.experiment.run()

        self.assertEqual(len(os.listdir(os.path.join(self.output_dir, 'raw_data'))), 2)
        for run_dir in os.listdir(os.path.join(self.output_dir, 'raw_data')):
            run = os.path.join(self.output_dir, 'raw_data', run_dir)
            for f in ['applied_action', 'current_waypoint', 'depth_scan', 'odometry', 'ros_expert', 'termination']:
                self.assertTrue(os.path.isfile(os.path.join(run, f)))
                if f != 'termination':
                    with open(os.path.join(run, f), 'r') as data_file:
                        self.assertTrue(len(data_file.readlines()) > 50)
            # check forward_camera
            self.assertTrue(len(os.listdir(os.path.join(run, 'forward_camera'))) > 50)
            img_numbers = [float(img.split('.')[0]) for img in sorted(os.listdir(os.path.join(run, 'forward_camera')))]
            distances = [img_numbers[i+1] - img_numbers[i] for i in range(len(img_numbers)-1)]
            self.assertTrue(max(distances) < 500)  # asserting the largest delay < 2 FPS or 500ms

    def tearDown(self) -> None:
        self.experiment.shutdown()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
