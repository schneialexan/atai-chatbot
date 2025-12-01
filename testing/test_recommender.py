import unittest
import os
import sys
import time

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import App

class TestMovieRecommendations(unittest.TestCase):
    """Tests for the movie recommendation system (knowledge graph + embeddings + ratings)."""

    def setUp(self):
        """Set up test environment."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(base_dir, "..", "dataset", "store", "graph_cache.pkl") # Correct the path to be relative to the project root
        self.embeddings_path = os.path.join(base_dir, "..", "dataset", "embeddings")

        # Single App instance with lazy initialization
        self.app = App(
            dataset_path=self.dataset_path,
            embeddings_path=self.embeddings_path,
            preload_strategy="none"
        )
        self.startTime = time.time()

    def tearDown(self):
        """Measure runtime per test."""
        t = time.time() - self.startTime
        print(f"{self.id()}: {t:.4f}s")
    
    def test_recommender_questions(self):
        """Test recommender questions."""
        test_cases = [{
                "question": "Recommend movies similar to La Dolce Vita and The Voice of the Moon.",
                "expected": "Fellini"
            },
            {
                "question": "Given that I like The Lion King, Pocahontas, and Beauty and the Beast, can you recommend some movies?",
                "expected": "Disney"
            },
            {
                "question": "Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween.",
                "expected": "Horror"
            },
            {
                "question": "Recommend movies similar to Hamlet and Othello.",
                "expected": "Shakespear"
            },
            {
                "question": "I like Singin' in the Rain, and Moulin Rouge. What other movies would you recommend for me to watch?",
                "expected": "musical"
            },
            {
                "question": "I really enjoyed Chicago, Memoirs of a Geisha, and Alice in Wonderland. Can you recommend me some similar movies?",
                "expected": "Colleen Atwood"
            },
            {
                "question": "What other movies in Japanese do you recommend? I liked Twin Sisters of Kyoto.",
                "expected": "Tōru Takemitsu"
            },
            {
                "question": "Can you recommend some biographical movies given that I like Meryl Streep?",
                "expected": "Julia"
            },
            {
                "question": "I really enjoy movies featuring Tom Hanks, what would you recommend?",
                "expected": "Forrest Gump"
            },
            {
                "question": "Can you give me some films connected to Leonardo DiCaprio?",
                "expected": "Inception"
            }
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
                result = self.app.get_answer(case["question"], mode=3)
                # add icon to print
                print(f"A: {result}")
                print(100*"=")
                if isinstance(case["expected"], list):
                    for expected_item in case["expected"]:
                        self.assertIn(expected_item, str(result))
                else:
                    self.assertIn(case["expected"], str(result))