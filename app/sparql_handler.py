import pickle
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

class LocalSPARQL:
    def __init__(self, dataset_path: str):
        """
        Initialize the SPARQL handler with a given RDF dataset
        """
        self.graph = Graph()
        try:
            print(f"[SPARQL] Loading RDF graph from: {dataset_path}")
            with open(dataset_path, "rb") as f:
                self.graph: Graph = pickle.load(f)
            print(f"[SPARQL] Loaded {len(self.graph)} triples.")
        except Exception as e:
            raise RuntimeError(f"Failed to load RDF dataset from {dataset_path}: {e}")

    @property
    def schema(self) -> str:
        """
        Returns a string representation of a few triples from the graph.
        This is used to provide context to LLMs.
        """
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"
        results: SPARQLResult = self.graph.query(query)
        # n3() method provides a Turtle-like representation of the term
        return "\n".join([f"{s.n3()} {p.n3()} {o.n3()}." for s, p, o in results])

    def get_all_properties(self):
        """
        Returns a list of all unique properties in the graph.
        """
        query = "SELECT DISTINCT ?p WHERE { ?s ?p ?o }"
        results = self.query(query)
        return [row['p'] for row in results if isinstance(results, list)]

    def get_all_entities(self):
        """
        Returns a list of all unique entities (subjects and objects) and their labels in the graph.
        """
        query = """
            SELECT DISTINCT ?entity ?label WHERE {
                { ?entity rdfs:label ?label . }
                UNION
                { ?entity a ?class . ?entity rdfs:label ?label . }
                FILTER(LANG(?label) = '' || LANGMATCHES(LANG(?label), 'en'))
            }
        """
        results = self.query(query)
        return results if isinstance(results, list) else []

    def _execute_raw_query(self, sparql_query: str) -> SPARQLResult:
        """
        Executes a SPARQL query and returns the raw rdflib result object.
        """
        return self.graph.query(sparql_query) 

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
            return {"error": "I could not find any results for your query. Please try again with a different query!"}
        return parsed_results
