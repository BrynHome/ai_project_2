import unittest
from preprocessor import clean_text

class TestTextPreprocessing(unittest.TestCase):

    def test_clean_text(self):

        filtered = clean_text(text="the. hello! this is cool. ☎✈♕ ☎✈♕.")
        self.assertEqual(filtered, "hello cool")

        filtered = clean_text(text="我们. hello! this is cool. ☎✈♕ ☎✈♕.")
        self.assertEqual(filtered, "hello cool")

        filtered = clean_text(text="")
        self.assertEqual(filtered, "")


if __name__ == '__main__':
    unittest.main()
