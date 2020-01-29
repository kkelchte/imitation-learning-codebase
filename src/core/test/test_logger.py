import unittest
import os
import shutil

from datetime import datetime

import src.core.logger as logger


class TestLogger(unittest.TestCase):

    def setUp(self) -> None:
        self.TEST_DIR = f'test-{datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M")}'
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

    def test_normal_usage(self):
        current_logger = logger.get_logger(name=os.path.basename(__file__),
                                           output_path=self.TEST_DIR)
        current_logger.debug(f'debug message')
        current_logger.info(f'info message')
        current_logger.warning(f'warning message')
        current_logger.error(f'error message')

        f = open(os.path.join(self.TEST_DIR, 'logfile'), 'r')
        log_lines = f.readlines()
        self.assertEqual(len(log_lines), 4)
        f.close()

    def test_quite(self):
        current_logger = logger.get_logger(name=os.path.basename(__file__),
                                           output_path=self.TEST_DIR,
                                           quite=True)
        current_logger.debug(f'debug message')
        current_logger.info(f'info message')
        current_logger.warning(f'warning message')
        current_logger.error(f'error message')

        f = open(os.path.join(self.TEST_DIR, 'logfile'), 'r')
        log_lines = f.readlines()
        self.assertEqual(len(log_lines), 4)
        f.close()

    def test_cprint(self):
        current_logger = logger.get_logger(name=os.path.basename(__file__),
                                           output_path=self.TEST_DIR,
                                           quite=True)
        logger.cprint('HELP', current_logger)

        f = open(os.path.join(self.TEST_DIR, 'logfile'), 'r')
        log_line = f.readlines()[0].strip()
        print(log_line)
        self.assertTrue('HELP' in log_line)
        f.close()

    def tearDown(self):
        os.remove(os.path.join(self.TEST_DIR, 'logfile'))
        shutil.rmtree(self.TEST_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
