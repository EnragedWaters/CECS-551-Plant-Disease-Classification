import unittest

class DeliberateFailure(unittest.TestCase):
    def test_merge_guard(self):
        self.fail("Intentional verification failure; never merge this branch")
