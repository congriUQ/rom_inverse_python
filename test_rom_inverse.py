import unittest
import torch
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class SampleMaterialPropertyField(unittest.TestCase):
    """
    Tests sampling of material property field
    """
    def test_squared_exponential_kernel(self):
        pass



if __name__ == '__main__':
    unittest.main()
