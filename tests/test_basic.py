"""Basic tests for the AllergyP package."""

import unittest
import allergyp

class TestBasic(unittest.TestCase):
    def test_version(self):
        """Test that the version is a string."""
        self.assertIsInstance(allergyp.__version__, str)

if __name__ == "__main__":
    unittest.main() 