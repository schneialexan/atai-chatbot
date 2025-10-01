# main.py
from app.core import App

if __name__ == "__main__":
    endpoint_url = "https://query.wikidata.org/sparql"

    # Mode 1: raw SPARQL (example that exists on Wikidata: list 5 cats)
    message_sparql = """
    SELECT ?item ?itemLabel
    WHERE {
      ?item wdt:P31 wd:Q146.   # instance of: cat
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 5
    """

    app = App(endpoint_url, mode=1)
    print("Mode 1 (raw SPARQL):", app.post_message(message_sparql))

    # Mode 2: factual via LLM->SPARQL or property mapping
    app = App(endpoint_url, mode=2)
    # This is an example known to be in Wikidata
    print("Mode 2 (factual):", app.post_message("Who is the director of The Bridge on the River Kwai?"))
    # expected answer: David Lean

    # Mode 2: embedding-style (if no LLM or property mapping gives nothing, fallback to embeddings)
    print("Mode 2 (embedding/fallback):", app.post_message("What is the genre of Good Neighbors?"))
    # expected answer: drama, action-drama etc.

    # Mode 3: recommendation
    #app = App(endpoint_url, mode=3)
    #print("Mode 3 (recommendation):", app.post_message("I like The Lion King, Pocahontas, and The Beauty and the Beast. Recommend some movies."))
    # expected answer: 2d disney movies

    # Mode 4: auto-detect (example: SPARQL)
    app = App(endpoint_url, mode=4)
    print("Mode 4 (auto SPARQL):", app.post_message(message_sparql))
    print("Mode 4 (auto QA):", app.post_message("Who directed Star Wars: Episode VI - Return of the Jedi?"))
