import unittest
import os
import sys
import time

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import App

class TestMultimediaHandler(unittest.TestCase):
    """Tests for the multimedia handler (image/IMDB links)."""

    def setUp(self):
        """Set up the test environment."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(base_dir, "..", "dataset", "store", "graph_cache.pkl")
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
        test_cases = [
            {
                "question": "Show me a picture of Halle Berry.",
                "expected": "image:0344/9aLI0LSi7cbieyiskOdsBaneKmp"
            },
            {
                "question": "What does Denzel Washington look like?",
                "expected": "image:0278/393wX9AGWpseVqojQDPLy3bTBia"
            },
            {
                "question": "Let me know what Sandra Bullock looks like.",
                "expected": "image:0249/hPHGKPAWZ8gArYXMk225rrYPoyJ"
            },
            {
                "question": "Let me know how the cover of Interstellar looks like.",
                "expected": "image:0055/gEU2QniE6E77NI6lCU6MxlNBvIx"
            },
            {
                "question": "Let me know how the cover of Forrest Gump looks like.",
                "expected": "image:0077/zxzYh2YtgypKrijVE0OuIyEgwdT"
            },
            {
                "question": "Show me the picture of the film The Hobbit: The Desolation of Smaug!",
                "expected": "image:0147/xQYiXsheRCDBA39DOrmaw1aSpbk"
            },
        ]

        # Ensure test_cases is a list
        if not isinstance(test_cases, list):
            self.fail(f"test_cases is not a list: {type(test_cases)} - {test_cases}")
        
        for idx, case in enumerate(test_cases):
            # Ensure case is a dictionary
            if not isinstance(case, dict):
                self.fail(f"Test case {idx} is not a dictionary: {type(case)} - {case}")
            with self.subTest(question=case["question"]):
                print(100*"=")
                print(f"Q: {case['question']}")
                result = self.app.get_answer(case["question"], mode=4)
                print(f"A: {result}")
                print(100*"=")
                self.assertEqual(result, case["expected"])

if __name__ == '__main__':
    unittest.main()

