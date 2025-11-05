import utils
import unittest


class get_list_depth(unittest.TestCase):

    def test(self):
        self.assertEqual(utils.get_list_depth(1234), 0)
        self.assertEqual(utils.get_list_depth([]), 1)
        self.assertEqual(utils.get_list_depth([1, 2, 3]), 1)
        self.assertEqual(utils.get_list_depth([1, [2, 3]]), 2)
        self.assertEqual(utils.get_list_depth([[1], [2, 3]]), 2)
