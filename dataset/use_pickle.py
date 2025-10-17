import os
import pickle
import time
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

# Paths
CACHE_DIR = "dataset/store"
CACHE_PATH = os.path.join(CACHE_DIR, "graph_cache.pkl")

# Ensure folder exists
os.makedirs(CACHE_DIR, exist_ok=True)

start_load = time.time()
g = Graph()

# Step 2: Load cached graph
print("[INFO] Loading graph from pickle cache...")
with open(CACHE_PATH, "rb") as f:
    g: Graph = pickle.load(f)
load_time = time.time() - start_load
print(f"[INFO] Graph loaded with {len(g)} triples in {load_time:.2f} seconds.")

# Step 3: Define SPARQL query
sparql_query = """
PREFIX ddis: <http://ddis.ch/atai/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?movieLabel ?movieltem WHERE {
  ?movieltem wdt:P31 wd:Q11424.
  ?movieltem ddis:rating ?rating.
  ?movieltem rdfs:label ?movieLabel.
}
ORDER BY DESC(?rating)
LIMIT 1
"""

# Step 4: Run SPARQL query and measure time
print("[INFO] Executing SPARQL query...")
start_query = time.time()
results: SPARQLResult = g.query(sparql_query)

# Step 5: Print results
parsed_results = []
for row in results:
    row_dict = {}
    for var, val in row.asdict().items():
        row_dict[var] = str(val) if val else None
    parsed_results.append(row_dict)
query_time = time.time() - start_query
print(f"[RESULT] Query executed in {query_time:.2f} seconds.")
print(parsed_results)
g.close()
