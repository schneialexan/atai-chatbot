import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Set, Any, Tuple
from collections import defaultdict
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import requests

from app.llm.llm_handler import LLMHandler
from app.llm.prompt_manager import PromptManager
from app.sparql_handler import LocalSPARQL

CACHED_PROPERTIES_FILE = "app/cached_properties.json"
CACHED_ENTITIES_FILE = "app/cached_entities.json"

class QAHandler:
    def __init__(self, llm_handler: LLMHandler, sparql_handler: LocalSPARQL, ner_handler: LLMHandler, embedding_handler: LLMHandler = None, dataset_path: str = "dataset", embeddings_path: str = "dataset/store/embeddings"):
        self.llm_handler = llm_handler
        self.sparql_handler = sparql_handler
        self.ner_handler = ner_handler
        self.embedding_handler = embedding_handler
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.prompt_manager = PromptManager()
        self.properties = self._get_properties()
        self.property_synonym_map = self._load_property_synonyms()
        self.entities = self._get_entities()
        self.entity_embeddings = None
        self.entity_ids = None
        self.relation_embeddings = None
        self.relation_ids = None
        self._load_embeddings()

    def _get_properties(self) -> Dict[str, str]:
        """Gets all properties from the knowledge graph and their labels, with caching."""
        if os.path.exists(CACHED_PROPERTIES_FILE):
            with open(CACHED_PROPERTIES_FILE, "r", encoding="utf-8") as f:
                print(f"Loading cached properties from {CACHED_PROPERTIES_FILE}")
                return json.load(f)

        properties = {}
        print("Querying knowledge graph for all properties...")
        for prop_uri in self.sparql_handler.get_all_properties():
            label = self._get_label_for_uri(prop_uri)
            if label:
                properties[label.lower()] = prop_uri  # Store label in lowercase for easier matching
        
        with open(CACHED_PROPERTIES_FILE, "w", encoding="utf-8") as f:
            json.dump(properties, f, indent=4, ensure_ascii=False)
        print(f"Cached properties saved to {CACHED_PROPERTIES_FILE}")
        return properties

    def _get_entities(self) -> Dict[str, str]:
        """Gets all entities from the knowledge graph and their labels, with caching."""
        if os.path.exists(CACHED_ENTITIES_FILE):
            with open(CACHED_ENTITIES_FILE, "r", encoding="utf-8") as f:
                print(f"Loading cached entities from {CACHED_ENTITIES_FILE}")
                return json.load(f)

        entities = {}
        print("Querying knowledge graph for all entities...")
        for entity_data in self.sparql_handler.get_all_entities():
            if isinstance(entity_data, dict) and 'entity' in entity_data and 'label' in entity_data:
                entity_uri = entity_data['entity']
                label = entity_data['label']
                if label:
                    entities[label.lower()] = entity_uri  # Store label in lowercase for easier matching
                
        with open(CACHED_ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=4, ensure_ascii=False)
        print(f"Cached entities saved to {CACHED_ENTITIES_FILE}")
        return entities

    def _load_property_synonyms(self) -> Dict[str, List[str]]:
        """Loads property synonyms from the JSON file and returns the synonym mapping."""
        synonym_data = {}
        try:
            with open("app/property_synonyms.json", "r", encoding="utf-8") as f:
                synonym_data = json.load(f)
        except FileNotFoundError:
            logging.warning("property_synonyms.json not found. Continuing without synonyms.")
            return {}

        return synonym_data

    def _load_embeddings(self):
        """Load pre-computed embeddings from the dataset."""
        try:
            if os.path.exists(self.embeddings_path):
                print("Loading entity embeddings...")
                self.entity_embeddings = np.load(os.path.join(self.embeddings_path, "entity_embeds.npy"))
                self.entity_ids = np.loadtxt(os.path.join(self.embeddings_path, "entity_ids.del"), dtype=str)
                
                print("Loading relation embeddings...")
                self.relation_embeddings = np.load(os.path.join(self.embeddings_path, "relation_embeds.npy"))
                self.relation_ids = np.loadtxt(os.path.join(self.embeddings_path, "relation_ids.del"), dtype=str)
                
                print(f"Loaded {len(self.entity_ids)} entity embeddings and {len(self.relation_ids)} relation embeddings")
            else:
                print("Warning: Embeddings directory not found. Embedding fallback will not be available.")
        except Exception as e:
            print(f"Warning: Could not load embeddings: {e}. Embedding fallback will not be available.")

    def answer(self, question: str):
        # 1. Extract entities from the question
        entities = self._extract_entities(question)
        if not entities:
            return {"error": "No entities found in the question."}
        print(f"Extracted entities: {entities}")
        # 2. Find the URIs for the extracted entities
        entity_uris = self._find_entity_uris(entities)
        if not entity_uris:
            return {"error": "Could not find any of the entities in the knowledge graph."}
        print(f"Found entity URIs: {entity_uris}")
        # 3. Identify the property the user is asking about
        property_uri = self._identify_property(question)
        if not property_uri:
            return {"error": "Could not determine the property you are asking about."}
        print(f"Identified property URI: {property_uri}")
        # 4. Construct and execute the SPARQL query
        # For now, we'll assume the first entity found is the main subject
        main_entity_uri = list(entity_uris.values())[0]
        print(f"Main entity URI: {main_entity_uri}")
        # Convert property URI to wdt: format if it's a Wikidata property
        if property_uri.startswith('http://www.wikidata.org/prop/direct/'):
            prop_id = property_uri.split('/')[-1]
            property_predicate = f"wdt:{prop_id}"
        else:
            property_predicate = f"<{property_uri}>"
        
        sparql_query = f"""
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            SELECT ?answerLabel ?answerItem WHERE {{
                <{main_entity_uri}> {property_predicate} ?answerItem .
                OPTIONAL {{ ?answerItem rdfs:label ?answerLabel .
                           FILTER(LANG(?answerLabel) = '' || LANGMATCHES(LANG(?answerLabel), 'en')) }}
            }}
        """
        print(f"DEBUG: SPARQL Query: {sparql_query}")
        results = self.sparql_handler.query(sparql_query)
        print(f"DEBUG: SPARQL Results: {results}")

        # 5. Format the answer
        return self._format_answer(results, question)

    def _extract_entities(self, question: str) -> List[str]:
        """Extracts entities from the question using cached entities first, then explicit quotes and NER as fallback."""
        entities = []
        processed_question = question
        question_lower = question.lower()
        
        # 1. Try to find entities from cached entities in the question
        cached_entity_labels = list(self.entities.keys())
        
        # TODO: Cosine similarity to find entities

        # Sort by length (longest first) to match longer entity names before shorter ones
        sorted_entities = sorted(cached_entity_labels, key=len, reverse=True)
        
        # TODO: Extract entities for full words only, not partial matches

        for entity_label in sorted_entities:
            # Check if this entity label appears in the question (case-insensitive)
            if entity_label in question_lower:
                # Extract the original case version from the question
                pattern = re.compile(re.escape(entity_label), re.IGNORECASE)
                matches = pattern.findall(question)
                for match in matches:
                    if match.lower() not in [e.lower() for e in entities]:
                        entities.append(match)
                        # Remove the matched entity from the question to avoid duplicate extraction
                        processed_question = processed_question.replace(match, "", 1)
                        question_lower = processed_question.lower()
        
        # 2. Extract entities explicitly enclosed in single quotes
        quoted_entities = re.findall(r"'([^']+)'", processed_question)
        for q_entity in quoted_entities:
            if q_entity.lower() not in [e.lower() for e in entities]:
                entities.append(q_entity)
                # Remove the quoted entity from the question to avoid confusing the NER model
                processed_question = processed_question.replace(f"'{q_entity}'", q_entity)
        
        if len(entities) == 0:
            # 3. Run NER on the (potentially preprocessed) question as fallback
            response = self.ner_handler.generate_ner_response(processed_question, aggregation_strategy="simple")
            if response['success']:
                for entity in response['content']:
                    # Only add if not already extracted and not just punctuation
                    if (entity['word'].lower() not in [e.lower() for e in entities] and 
                        re.search(r'[a-zA-Z0-9]', entity['word'])):
                        entities.append(entity['word'])
        
        return entities

    def _find_entity_uris(self, entities: List[str]) -> Dict[str, str]:
        """Finds the URIs for the given entities using cached entities first, then SPARQL fallback."""
        entity_uris = {}
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # 1. Try exact match in cached entities
            if entity_lower in self.entities:
                entity_uris[entity] = self.entities[entity_lower]
                continue
            
            # 2. Try fuzzy matching in cached entities
            cached_entity_labels = list(self.entities.keys())
            best_match, score = process.extractOne(entity_lower, cached_entity_labels)
            
            if score >= 90:  # High confidence threshold for fuzzy matching
                entity_uris[entity] = self.entities[best_match]
                continue
            
            # 3. Try partial matching in cached entities
            for cached_label, uri in self.entities.items():
                if entity_lower in cached_label or cached_label in entity_lower:
                    entity_uris[entity] = uri
                    break
            
            if entity in entity_uris:
                continue
            
            # 4. Fallback to SPARQL queries (original implementation)
            # Try exact match first
            query = f"""
                SELECT ?s WHERE {{
                    ?s rdfs:label "{entity}"@en .
                }}
            """
            results = self.sparql_handler.query(query)
            if isinstance(results, list) and len(results) > 0 and 's' in results[0]:
                entity_uris[entity] = results[0]['s']
                continue
            
            # Try case-insensitive match
            query = f"""
                SELECT ?s WHERE {{
                    ?s rdfs:label ?label .
                    FILTER(LCASE(?label) = LCASE("{entity}"))
                }}
            """
            results = self.sparql_handler.query(query)
            if isinstance(results, list) and len(results) > 0 and 's' in results[0]:
                entity_uris[entity] = results[0]['s']
                continue
            
            # Try partial match
            query = f"""
                SELECT ?s WHERE {{
                    ?s rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(?label), LCASE("{entity}")))
                }}
            """
            results = self.sparql_handler.query(query)
            if isinstance(results, list) and len(results) > 0 and 's' in results[0]:
                entity_uris[entity] = results[0]['s']
        
        return entity_uris

    def _identify_property(self, question: str, fuzzy_threshold: int = 80) -> str:
        """Identifies the property the user is asking about."""
        question_lower = question.lower()
        processed_question = question_lower

        # 1. Replace synonyms in the question with their canonical property names
        # We need to be careful with order to avoid partial matches (e.g., 'cast' before 'cast member')
        # Sort synonyms by length in descending order to match longer phrases first
        
        # First, collect all synonyms and sort by length
        all_synonyms = []
        for prop_name, synonyms_list in self.property_synonym_map.items():
            for syn in synonyms_list:
                all_synonyms.append((syn, prop_name))
        
        # Sort by synonym length (longest first) to avoid partial matches
        all_synonyms.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Use word boundaries to ensure we only replace whole words/phrases
        for syn, prop_name in all_synonyms:
            # Create a regex pattern that matches the synonym as a whole word or phrase
            # Use word boundaries for single words, or exact phrase matching for multi-word phrases
            if ' ' in syn:
                # Multi-word phrase - use exact phrase matching
                pattern = re.compile(re.escape(syn), re.IGNORECASE)
                if pattern.search(processed_question):
                    print(f"DEBUG: Found multi-word synonym '{syn}' -> '{prop_name}'")
                    processed_question = pattern.sub(prop_name, processed_question, count=1)
                    break
            else:
                # Single word - use word boundaries
                pattern = re.compile(r'\b' + re.escape(syn) + r'\b', re.IGNORECASE)
                if pattern.search(processed_question):
                    print(f"DEBUG: Found single-word synonym '{syn}' -> '{prop_name}'")
                    processed_question = pattern.sub(prop_name, processed_question, count=1)
                    break

        print(f"Processed question: {processed_question}")
        
        # 2. Fuzzy string matching against the canonical property names
        canonical_property_names = list(self.properties.keys())
        best_match, score = process.extractOne(processed_question, canonical_property_names)
        
        print(f"DEBUG: Fuzzy matching - Best match: '{best_match}' with score: {score}")
        
        if score > fuzzy_threshold:
            return self.properties.get(best_match)
        else:
            print(f"ERROR: No property found for question: {question}")
            return None

    def _format_answer(self, results: Dict[str, Any], question: str) -> str:
        """Formats the answer for the user using LLM with prompt for natural language formatting."""
        if isinstance(results, dict) and "error" in results:
            print(f"ERROR: {results['error']}")

        # Handle different types of answers
        answers = []
        for res in results:
            if 'answerLabel' in res and res['answerLabel']:
                # Entity with a label
                answers.append(res['answerLabel'])
            elif 'answerItem' in res and res['answerItem']:
                # Check if it's a literal value or an entity
                answer_item = res['answerItem']
                if answer_item.startswith('http'):
                    # It's a URI - try to get its label
                    label = self._get_label_for_uri(answer_item)
                    if label != answer_item:  # If we got a proper label
                        answers.append(label)
                    else:
                        answers.append(answer_item)
                else:
                    # It's a literal value (like a date)
                    answers.append(str(answer_item))
        
        if not answers:
            return "I could not find an answer to your question in my knowledge graph.. I'm sorry about that."

        # Use LLM to format the answer naturally
        try:
            # Prepare the raw data for the LLM
            raw_data = {
                "answers": answers,
                "count": len(answers)
            }
            
            # Get the prompt for natural language formatting
            prompt = self.prompt_manager.get_prompt(
                "result_to_natural_language",
                question=question,
                raw_data=json.dumps(raw_data, indent=2, ensure_ascii=False)
            )
            
            # Generate a natural language response using the LLM
            response = self.llm_handler.generate_response(prompt)
            
            if response['success']:
                return "Based on my knowledge graph: " + response['content']
            else:
                # Fallback to simple formatting if LLM fails
                return "Based on my knowledge graph: " + ", ".join(answers)
                
        except Exception as e:
            print(f"Error formatting answer with LLM: {e}")
            # Fallback to simple formatting if LLM fails
            return "Based on my knowledge graph: " + ", ".join(answers)

    def _get_label_for_uri(self, uri: str) -> str:
        """Gets the label for a given URI."""
        # TODO: implement wikidata label lookup as fallback
        query = f"""
            SELECT ?label WHERE {{
                <{uri}> rdfs:label ?label .
                FILTER(LANG(?label) = '' || LANGMATCHES(LANG(?label), 'en'))
            }}
        """
        results = self.sparql_handler.query(query)
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
            response = requests.get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={id}&format=json&languages=en", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'entities' in data and id in data['entities']:
                    print(f"Found label in Wikidata: {data['entities'][id]['labels']['en']['value']}")
                    return data['entities'][id]['labels']['en']['value']
                else:
                    print(f"ERROR: No label found for ID in Wikidata: {id}")
                    return uri
            else:
                print(f"ERROR: Failed to query Wikidata for ID: {id}")
                return uri
