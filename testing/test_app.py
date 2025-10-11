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
        self.dataset_path = os.path.join(base_dir, "..", "dataset", "graph.nt") # Correct the path to be relative to the project root
        # Single App instance with lazy initialization
        self.app = App(self.dataset_path, preload_strategy="none")
        self.startTime = time.time()

    def tearDown(self):
        """Tear down the test environment."""
        t = time.time() - self.startTime
        print(f"{self.id()}: {t:.4f}s")

    def test_sparql_queries(self):
        """Test SPARQL queries from the intermediate evaluation."""
        # Query 1
        query = '''
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        SELECT ?movieLabel ?movieltem WHERE {
          ?movieltem wdt:P31 wd:Q11424.
          ?movieltem ddis:rating ?rating.
          ?movieltem rdfs:label ?movieLabel.
        }
        ORDER BY DESC(?rating)
        LIMIT 1
        '''
        expected_string = 'Acidulous Midtime Shed'
        expected_id = 'Q10850238456619979'
        result = self.app.get_answer(query, mode=1)
        self.assertIn(expected_string, [res.get('movieLabel') for res in result])
        # Extract IDs from full URLs for comparison
        movie_ids = [res.get('movieltem', '').split('/')[-1] for res in result]
        self.assertIn(expected_id, movie_ids)

        # Query 2
        query = '''
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        SELECT ?directorLabel ?directorItem WHERE {
          ?movieltem rdfs:label "The Bridge on the River Kwai".
          ?movieltem wdt:P57 ?directorItem.
          ?directorItem rdfs:label ?directorLabel.
        }
        '''
        expected_string = 'David Lean'
        expected_id = 'Q55260'
        result = self.app.get_answer(query, mode=1)
        self.assertIn(expected_string, [res.get('directorLabel') for res in result])
        # Extract IDs from full URLs for comparison
        director_ids = [res.get('directorItem', '').split('/')[-1] for res in result]
        self.assertIn(expected_id, director_ids)

        # Query 3
        query = '''
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        SELECT ?genreLabel ?genreltem WHERE {
          ?movieltem rdfs:label "Shoplifters".
          ?movieltem wdt:P136 ?genreltem.
          ?genreltem rdfs:label ?genreLabel.
        }
        '''
        expected_string = 'drama film'
        expected_id = 'Q130232'
        result = self.app.get_answer(query, mode=1)
        self.assertIn(expected_string, [res.get('genreLabel') for res in result])
        # Extract IDs from full URLs for comparison
        genre_ids = [res.get('genreltem', '').split('/')[-1] for res in result]
        self.assertIn(expected_id, genre_ids)

        # Query 4
        query = '''
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        SELECT ?producerLabel ?producerltem WHERE {
          ?movieltem rdfs:label "French Kiss".
          ?movieltem wdt:P162 ?producerltem .
          ?producerltem rdfs:label ?producerLabel.
        }
        '''
        result = self.app.get_answer(query, mode=1)
        self.assertIsInstance(result, list)
        producers = [res.get('producerLabel') for res in result]
        # Extract IDs from full URLs for comparison
        producers_ids = [res.get('producerltem', '').split('/')[-1] for res in result]
        self.assertIn('Tim Bevan', producers)
        self.assertIn('Meg Ryan', producers)
        self.assertIn('Eric Fellner', producers)
        self.assertIn('Q1473065', producers_ids)
        self.assertIn('Q167498', producers_ids)
        self.assertIn('Q1351291', producers_ids)

        # Query 5
        query = '''
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>
        SELECT ?movieLabel ?movieltem WHERE {
          ?movieltem wdt:P31 wd:Q11424.
          ?movieltem wdt:P495 ?countryltem.
          ?countryltem rdfs:label "South Korea".
          ?movieltem wdt:P166 ?awardItem.
          ?awardItem rdfs:label "Academy Award for Best Picture".
          ?movieltem rdfs:label ?movieLabel.
        }
        '''
        expected_string = 'Parasite'
        expected_id = 'Q61448040'
        result = self.app.get_answer(query, mode=1)
        self.assertIn(expected_string, [res.get('movieLabel') for res in result])
        # Extract IDs from full URLs for comparison
        movie_ids = [res.get('movieltem', '').split('/')[-1] for res in result]
        self.assertIn(expected_id, movie_ids)


    def test_factual_questions(self):
        """Test factual questions from the final evaluation."""
        question = "Who is the director of Good Will Hunting?"
        expected = "Gus Van Sant"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result))

        question = "Who directed The Bridge on the River Kwai?"
        expected = "David Lean"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result))

        question = "Who is the director of Star Wars: Episode VI - Return of the Jedi?"
        expected = "Richard Marquand"
        result = self.app.get_answer(question, mode=2)
        self.assertIn(expected, str(result))

    def test_embedding_questions(self):
        """Test embedding questions from the final evaluation."""
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
        """Test multimedia questions from the final evaluation."""
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
        """Test recommendation questions from the final evaluation."""
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