import unittest
import os
import sys
import time

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import App

class TestApp(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(base_dir, "..", "dataset", "store", "graph_cache.pkl") # Correct the path to be relative to the project root
        self.embeddings_path = os.path.join(base_dir, "..", "dataset", "embeddings")
        # Single App instance with lazy initialization
        self.app = App(self.dataset_path, self.embeddings_path, preload_strategy="none")
        self.startTime = time.time()

    def tearDown(self):
        """Tear down the test environment."""
        t = time.time() - self.startTime
        print(f"{self.id()}: {t:.4f}s")

    def test_multimedia_questions(self):
        """Test multimedia questions."""
        question = "Show me a picture of Halle Berry."
        expected = "https://www.imdb.com/name/nm0000932"
        result = self.app.get_answer(question, mode=4)
        self.assertEqual(result, expected)

        question = "What does Denzel Washington look like?"
        expected = "https://www.imdb.com/name/nm0000243"
        result = self.app.get_answer(question, mode=4)
        self.assertEqual(result, expected)

        question = "Let me know what Sandra Bullock looks like."
        expected = "https://www.imdb.com/name/nm0000113"
        result = self.app.get_answer(question, mode=4)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()