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
        
    def test_embedding_questions(self):
        """Test embedding questions."""
        question = "Who is the screenwriter of The Masked Gang: Cyprus?"
        expected = "Cengiz Küçükayvaz"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result))

        question = "What is the MPAA film rating of Weathering with You?"
        expected = "PG-13"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result))

        question = "What is the genre of Good Neighbors?"
        expected = "drama"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result).lower())

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

    def test_recommendation_questions(self):
        """Test recommendation questions."""
        question = "Recommend movies similar to Hamlet and Othello."
        result = self.app.get_answer(question, mode=3)
        self.assertIsInstance(result, (str, dict))

        question = "Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"
        result = self.app.get_answer(question, mode=3)
        self.assertIsInstance(result, (str, dict))

        question = "Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween."
        result = self.app.get_answer(question, mode=3)
        self.assertIsInstance(result, (str, dict))


if __name__ == '__main__':
    unittest.main()