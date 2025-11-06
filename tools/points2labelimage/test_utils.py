import unittest

import points2label


class get_list_depth(unittest.TestCase):

    def test(self):
        self.assertEqual(points2label.get_list_depth(1234), 0)
        self.assertEqual(points2label.get_list_depth([]), 1)
        self.assertEqual(points2label.get_list_depth([1, 2, 3]), 1)
        self.assertEqual(points2label.get_list_depth([1, [2, 3]]), 2)
        self.assertEqual(points2label.get_list_depth([[1], [2, 3]]), 2)
