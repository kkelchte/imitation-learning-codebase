import unittest

from src.core.utils import camelcase_to_snake_format


class TestUtils(unittest.TestCase):

    def test_camelcase_to_snakeformat(self):
        self.assertEqual(camelcase_to_snake_format('ThisIsAFirstTest'),
                         'this_is_a_first_test')
        self.assertEqual(camelcase_to_snake_format('ThisIsA--SecondTest'),
                         'this_is_a--second_test')


if __name__ == '__main__':
    unittest.main()
