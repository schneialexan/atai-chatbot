import pickle
from typing import List, Dict
import rdflib
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib import URIRef
import requests
from rdflib.namespace import RDFS
from bs4 import BeautifulSoup

# define some prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
SCHEMA = rdflib.Namespace('http://schema.org/')

class LocalKnowledgeGraph:
    def __init__(self, dataset_path: str):
        """
        Initialize the LocalKnowledgeGraph with a given RDF dataset
        """
        self.graph = Graph()
        try:
            print(f"[LocalKnowledgeGraph] Loading RDF graph from: {dataset_path}")
            with open(dataset_path, "rb") as f:
                self.graph: Graph = pickle.load(f)
            print(f"[LocalKnowledgeGraph] Loaded {len(self.graph)} triples.")
        except Exception as e:
            raise RuntimeError(f"Failed to load RDF dataset from {dataset_path}: {e}")

    def get_all_properties(self):
        """
        Returns a list of all unique properties in the graph.
        """
        query = "SELECT DISTINCT ?p WHERE { ?s ?p ?o }"
        results = self.sparql_query(query)
        return [row['p'] for row in results if isinstance(results, list)]

    def get_all_entities(self):
        """
        Returns a list of all unique entity labels in the graph.
        """
        query = """
            SELECT DISTINCT ?label WHERE {
                { ?entity rdfs:label ?label . }
                UNION
                { ?entity a ?class . ?entity rdfs:label ?label . }
                FILTER(LANG(?label) = '' || LANGMATCHES(LANG(?label), 'en'))
            }
        """
        results = self.sparql_query(query)
        if isinstance(results, list):
            return [row['label'] for row in results if 'label' in row]
        elif isinstance(results, dict) and 'error' in results:
            print(f"[Entity Label Search] SPARQL query error: {results['error']}")
            return []
        else:
            return []

    def get_entity_property_labels(self, entity_uri: str) -> List[Dict[str, str]]:
        """
        Returns unique property URIs with their labels for a specific entity.
        Note: This returns only unique properties, not property-value pairs.
        
        Args:
            entity_uri: The URI of the entity to get properties for
            
        Returns:
            List of dictionaries with property URI and label:
            [{'property': 'property_uri', 'label': 'property_label'}]
        """
        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX schema: <http://schema.org/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT DISTINCT ?property ?label WHERE {{
                <{entity_uri}> ?property ?value .
                OPTIONAL {{ ?property rdfs:label ?label .
                           FILTER(LANG(?label) = '' || LANGMATCHES(LANG(?label), 'en')) }}
            }}
            ORDER BY ?property
        """
        
        results = self.sparql_query(query)
        if isinstance(results, list) and results:
            return results
        elif isinstance(results, dict) and 'error' in results:
            return []
        else:
            return []

    def get_entity_metadata_local(self, uri: str) -> Dict[str, List[str]]:
        """Fetches instance types and description for a local RDF entity."""
        uri = URIRef(uri)
        types = []
        descs = []
        
        # P31 (instance of)
        P31 = WDT.P31
        for obj in self.graph.objects(uri, P31):
            label = self.graph.value(obj, RDFS.label)
            if label:
                types.append(str(label).lower())

        # rdfs:comment or schema:description
        for p in [
            SCHEMA.description,
            RDFS.comment,
        ]:
            for obj in self.graph.objects(uri, p):
                descs.append(str(obj).lower())

        return {
            "types": list(set(types)),
            "description": " ".join(set(descs))
        }

    def sparql_query_raw(self, sparql_query: str) -> SPARQLResult:
        """
        Executes a SPARQL query and returns the raw rdflib result object.
        """
        return self.graph.query(sparql_query) 

    def sparql_query(self, sparql_query: str):
        """
        Executes a SPARQL query against the loaded graph.
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

    def get_label_for_uri(self, uri: str) -> str:
        """Gets the label for a given URI
        If no label is found in the knowledge graph, it looks up the label in Wikidata.
        """
        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE {{
                <{uri}> rdfs:label ?label .
                FILTER(LANG(?label) = '' || LANGMATCHES(LANG(?label), 'en'))
            }}
        """
        results = self.sparql_query(query)
        if isinstance(results, list) and len(results) > 0 and 'label' in results[0]:
            return results[0]['label']
        else:
            print(f"ERROR: No label found for URI in our knowledge graph, looking up in Wikidata: {uri}")
            # Extract id
            id = str(uri.split('/')[-1])
            # Add user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            # Try Wikidata API first
            response = requests.get(
                f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en&props=labels",
                headers=headers
            )
            label = None
            if response.status_code == 200:
                data = response.json()
                entity = data.get('entities', {}).get(id)
                if entity:
                    # Try English label
                    label = entity.get('labels', {}).get('en', {}).get('value')
                    if not label:
                        # Fallback: pick first available language
                        if 'labels' in entity and entity['labels']:
                            label = next(iter(entity['labels'].values()))['value']

            # Fallback: scrape HTML if no label found
            if not label:
                page_url = f"https://www.wikidata.org/wiki/{id}"
                response = requests.get(page_url, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    heading = soup.find('h1', id='firstHeading')
                    if heading:
                        label = heading.text.strip()
            if label:
                print(f"Found label in Wikidata: {label}")
                return label
            else:
                print(f"ERROR: No label found for ID: {id}")
                return id
