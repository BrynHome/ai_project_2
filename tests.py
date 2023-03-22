import unittest
from preprocessor import clean_text, detect_language

class TestTextPreprocessing(unittest.TestCase):

    def test_clean_text(self):

        filtered = clean_text(text="the. hello! this is cool. ☎✈♕ ☎✈♕.")
        self.assertEqual(filtered, "hello cool")

        filtered = clean_text(text="我们. hello! this is cool. ☎✈♕ ☎✈♕.")
        self.assertEqual(filtered, "hello cool")

        filtered = clean_text(text="")
        self.assertEqual(filtered, "")

    def test_detect_language(self):
        self.assertEqual("en", detect_language("Hello, this is awesome."))
        self.assertEqual("fr", detect_language("J'adore mon petit poisson."))

if __name__ == '__main__':
    unittest.main()
