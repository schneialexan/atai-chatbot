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

    def test_factual_questions(self):
        """Test factual questions."""
        test_cases = [
            {
                "question": "Please answer this question with a factual approach: From what country is the movie 'Aro Tolbukhin. En la mente del asesino'?",
                "expected": "Mexico"
            },
            {
                "question": "Please answer this question with a factual approach: Who is the screenwriter of 'Shortcut to Happiness'?",
                "expected": "Pete Dexter"
            },
            {
                "question": "Please answer this question with a factual approach: Who directed ‘Fargo’?",
                "expected": ["Ethan", "Joel"]
            },
            {
                "question": "Please answer this question with a factual approach: What genre is the movie 'Bandit Queen'?",
                "expected": ["drama", "biographical", "crime"]
            },
            {
                "question": "Please answer this question with a factual approach: When did the movie 'Miracles Still Happen' come out?",
                "expected": ["1974", "19"]
            },
            {
                "question": "Please answer this question with an embedding approach: Who is the director of ‘Apocalypse Now’?",
                "expected": "John Milius"
            },
            {
                "question": "Please answer this question with an embedding approach: Who is the screenwriter of ‘12 Monkeys’?",
                "expected": "Carol Florence"
            },
            {
                "question": "Please answer this question with an embedding approach: What is the genre of ‘Shoplifters’?",
                "expected": "comedy film"
            },
            {
                "question": "Please answer this question: Who is the director of ‘Good Will Hunting’?",
                "expected": "Gus Van Sant"
            },
            {
                "question": "Who is the director of Good Will Hunting?",
                "expected": "Gus Van Sant"
            },
            {
                "question": "Who is the director of the Beauty and the Beast?",
                "expected": "Bill Condon"
            },
            {
                "question": "Who directed The Bridge on the River Kwai?",
                "expected": "David Lean"
            },
            {
                "question": "Who is the director of Star Wars: Episode VI - Return of the Jedi?",
                "expected": "Richard Marquand"
            },
            {
                "question": "When was the movie 'Parasite' released?",
                "expected": "2019"
            },
            {
                "question": "Who are the main actors in 'The Bridge on the River Kwai'?",
                "expected": ["William Holden", "Alec Guinness"]
            },
            {
                "question": "Who is the director of Forrest Gump?",
                "expected": "Robert Zemeckis"
            },
            {
                "question": "Who directed The Godfather?",
                "expected": "Francis Ford Coppola"
            },
            {
                "question": "Who is the director of Jurassic Park?",
                "expected": "Steven Spielberg"
            },
            {
                "question": "Who is the director of The Longest Day?",
                "expected": ["Gerd Oswald", "Darryl F. Zanuck", "Ken Annakin", "Bernard Wicki"]
            },
            {
                "question": "What is the genre of Shoplifters?",
                "expected": "drama"
            },
            {
                "question": "Who acted in The Godfather?",
                "expected": ["Al Pacino", "Marlon Brando", "James Caan"]
            },
            {
                "question": "Who acted in Titanic?",
                "expected": ["Clifton Webb", "Barbara Stanwyck", "Robert Wagner", "Audrey Dalton"]
            },
            {
                "question": "Who is the main actor of Forrest Gump?",
                "expected": "Tom Hanks"
            },
            {
                "question": "Who is the screenwriter of 12 Monkeys?",
                "expected": ["David Peoples", "Chris Marker", "Janet Peoples"]
            },
            {
                "question": "Who made X-Men Beginnings?",
                "expected": "20th Century Studios"
            },
            {
                "question": "When was the movie The Godfather released?",
                "expected": "1972"
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
                result = self.app.get_answer(case["question"], mode=2)
                # add icon to print
                print(f"A: {result}")
                print(100*"=")
                if isinstance(case["expected"], list):
                    for expected_item in case["expected"]:
                        self.assertIn(expected_item, str(result))
                else:
                    self.assertIn(case["expected"], str(result))