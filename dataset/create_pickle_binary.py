"""
Run once to create a binary cache of the RDF graph.
"""
from rdflib import Graph
import os
import pickle

DATASET_PATH = "dataset/graph.nt"
CACHE_DIR = "dataset/store"
CACHE_PATH = os.path.join(CACHE_DIR, "graph_cache.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)

print("[INIT] Loading RDF graph...")
g = Graph()
g.parse(DATASET_PATH, format="nt")
print(f"[INIT] Loaded {len(g)} triples.")

print("[INIT] Serializing graph to pickle cache...")
with open(CACHE_PATH, "wb") as f:
    pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

print("[DONE] Cache created at", CACHE_PATH)
