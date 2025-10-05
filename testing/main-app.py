import os
import time
from app.core import App

def timed_app_init(dataset_path, mode):
    """Helper to time how long App initialization takes"""
    t0 = time.perf_counter()
    app = App(dataset_path, mode=mode)
    t1 = time.perf_counter()
    print(f"[Init] Mode {mode} initialization took {t1 - t0:.3f} seconds")
    return app

def timed_message(app, message, label=""):
    """Helper to time how long post_message takes"""
    t0 = time.perf_counter()
    result = app.post_message(message)
    t1 = time.perf_counter()
    print(f"[Query] {label} took {t1 - t0:.3f} seconds")
    return result

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "graph.nt")

    # Which movie has the highest user rating? 
    message_sparql = """
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
    
    SELECT ?lbl WHERE {
      ?movie wdt:P31 wd:Q11424 .
      ?movie ddis:rating ?rating .
      ?movie rdfs:label ?lbl .
    }
    ORDER BY DESC(?rating)
    LIMIT 1
    """

    # Which movie has the lowest user rating?
    message_sparql_2 = """
    PREFIX ddis: <http://ddis.ch/atai/>   
    PREFIX wd: <http://www.wikidata.org/entity/>   
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
    PREFIX schema: <http://schema.org/>   
    
    SELECT ?lbl WHERE {  
        ?movie wdt:P31 wd:Q11424 .  
        ?movie ddis:rating ?rating .  
        ?movie rdfs:label ?lbl .  
    }  
    ORDER BY ASC(?rating)   
    LIMIT 1
    """

    message_sparql_3 = """
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
      
    SELECT ?director WHERE {
        ?movie rdfs:label "Apocalypse Now"@en .
        ?movie wdt:P57 ?directorItem .
        ?directorItem rdfs:label ?director .
    }
    LIMIT 1
    """

    # Mode 1
    app = timed_app_init(dataset_path, mode=1)
    print("Mode 1 (raw SPARQL):", timed_message(app, message_sparql, "Raw SPARQL 1"))
    print("Mode 1 (raw SPARQL):", timed_message(app, message_sparql_2, "Raw SPARQL 2"))
    print("Mode 1 (raw SPARQL):", timed_message(app, message_sparql_3, "Raw SPARQL 3"))

    # Mode 2
    app = timed_app_init(dataset_path, mode=2)
    print("Mode 2 (factual):", timed_message(app, "Who is the director of The Bridge on the River Kwai?", "Factual"))
    print("Mode 2 (embedding/fallback):", timed_message(app, "What is the genre of Good Neighbors?", "Embedding"))

    # Mode 3
    app = timed_app_init(dataset_path, mode=3)
    print("Mode 3 (recommendation):", timed_message(app, "I like The Lion King, Pocahontas, and The Beauty and the Beast. Recommend some movies.", "Recommendation"))

    # Mode 4
    app = timed_app_init(dataset_path, mode=4)
    print("Mode 5 (multimedia):", timed_message(app, "Show me a picture of Halle Berry", "Multimedia"))

    # Mode 5
    app = timed_app_init(dataset_path, mode=5)
    print("Mode 4 (auto SPARQL):", timed_message(app, message_sparql, "Auto SPARQL"))
    print("Mode 4 (auto QA):", timed_message(app, "Who directed Star Wars: Episode VI - Return of the Jedi?", "Auto QA"))
