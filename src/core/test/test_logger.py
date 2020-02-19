import unittest
import os
import shutil
from glob import glob

import src.core.logger as logger
from src.core.utils import get_filename_without_extension


class TestLogger(unittest.TestCase):

    def setUp(self) -> None:
        self.TEST_DIR = f'test_dir/{get_filename_without_extension(__file__)}'
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

    def test_normal_usage(self):
        current_logger = logger.get_logger(name=os.path.basename(__file__),
                                           output_path=self.TEST_DIR)
        current_logger.debug(f'debug message')
        current_logger.info(f'info message')
        current_logger.warning(f'warning message')
        current_logger.error(f'error message')

        log_file = glob(os.path.join(self.TEST_DIR, 'log_files', '*'))[0]
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
            self.assertEqual(len(log_lines), 4)

    def test_multiple_loggers(self):
        logger_a = logger.get_logger(name='module_a',
                                     output_path=self.TEST_DIR)
        logger_b = logger.get_logger(name='module_b',
                                     output_path=self.TEST_DIR)
        logger.cprint('started', logger_a)
        logger.cprint('started', logger_b)
        log_files = glob(os.path.join(self.TEST_DIR, 'log_files', '*'))
        self.assertEqual(len(log_files), 2)

    def test_cprint(self):
        current_logger = logger.get_logger(name=os.path.basename(__file__),
                                           output_path=self.TEST_DIR,
                                           quite=True)
        logger.cprint('HELP', current_logger)

        log_file = glob(os.path.join(self.TEST_DIR, 'log_files', '*'))[0]
        with open(log_file, 'r') as f:
            log_line = f.readlines()[0].strip()
            print(log_line)
            self.assertTrue('HELP' in log_line)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
