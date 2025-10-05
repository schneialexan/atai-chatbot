# app/sparql_handler.py

import rdflib
from rdflib.plugins.sparql.processor import SPARQLResult

class LocalSPARQL:
    def __init__(self, dataset_path: str):
        """
        Initialize the SPARQL handler with a given RDF dataset (.nt, .ttl, etc.)
        """
        self.graph = rdflib.Graph()
        try:
            print(f"[SPARQL] Loading RDF graph from: {dataset_path}")
            # By default assume N-Triples (graph.nt). Can add format guessing later.
            self.graph.parse(dataset_path, format="nt")
            print(f"[SPARQL] Loaded {len(self.graph)} triples.")
        except Exception as e:
            raise RuntimeError(f"Failed to load RDF dataset from {dataset_path}: {e}")

    def query(self, sparql_query: str):
        """
        Execute a SPARQL query against the loaded graph.
        Returns results as list of dicts for easier use.
        """
        try:
            results: SPARQLResult = self.graph.query(sparql_query)
        except Exception as e:
            return {"error": f"SPARQL query failed: {e}"}

        # Convert rdflib results into Pythonic dicts
        parsed_results = []
        for row in results:
            row_dict = {}
            for var, val in row.asdict().items():
                row_dict[var] = str(val) if val else None
            parsed_results.append(row_dict)

        # Pretty return
        if not parsed_results:
            return {"message": "No results found."}
        return parsed_results
