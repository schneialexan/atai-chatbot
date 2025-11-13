import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz, process
from sentence_transformers import util
from sklearn.metrics import pairwise_distances

import rdflib
from rdflib.namespace import RDFS
import csv
import spacy
import difflib

from app.llm.llama_cpp_handler import LlamaCppHandler
from app.llm.transformer_handler import TransformerHandler
from app.llm.prompt_manager import PromptManager
from app.kg_handler import LocalKnowledgeGraph
from app.entity_extractor import EntityExtractor

CACHED_PROPERTIES_FILE = "app/cached_properties.json"

# define some prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
SCHEMA = rdflib.Namespace('http://schema.org/')

class QAHandler:
    def __init__(self, llm_handler: LlamaCppHandler, kg_handler: LocalKnowledgeGraph, embedding_handler: TransformerHandler, embeddings_path: str = "dataset/store/embeddings"):
        self.llm_handler = llm_handler
        self.kg_handler = kg_handler
        self.embedding_handler = embedding_handler
        self.embeddings_path = embeddings_path
        self.prompt_manager = PromptManager()
        
        self.properties = self._get_properties()
        self.property_names = self._get_property_names()
        self.property_synonym_map = self._load_property_synonyms()
        self.nlp = spacy.load("en_core_web_trf")
        self._load_embeddings_and_lookup_dictionaries()
        
        # Initialize EntityExtractor with embeddings lookup dictionaries if available
        self.entity_extractor = EntityExtractor(
            kg_handler=self.kg_handler,
            nlp=self.nlp,
            property_names=self.property_names,
            ent2lbl=self.ent2lbl if hasattr(self, 'ent2lbl') else None,
            all_entity_labels=self.all_entity_labels if hasattr(self, 'all_entity_labels') else None
        )

    def _get_properties(self) -> Dict[str, str]:
        """Gets all properties from the knowledge graph and their labels, with caching."""
        if os.path.exists(CACHED_PROPERTIES_FILE):
            with open(CACHED_PROPERTIES_FILE, "r", encoding="utf-8") as f:
                print(f"Loading cached properties from {CACHED_PROPERTIES_FILE}")
                return json.load(f)

        properties = {}
        print("Querying knowledge graph for all properties...")
        for prop_uri in self.kg_handler.get_all_properties():
            label = self.kg_handler.get_label_for_uri(prop_uri)
            if label:
                properties[label.lower()] = prop_uri  # Store label in lowercase for easier matching
        
        with open(CACHED_PROPERTIES_FILE, "w", encoding="utf-8") as f:
            json.dump(properties, f, indent=4, ensure_ascii=False)
        print(f"Cached properties saved to {CACHED_PROPERTIES_FILE}")
        return properties

    def _get_property_names(self) -> List[str]:
        """Gets all property names from the knowledge graph."""
        return list(self.properties.keys())

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

    def _load_embeddings_and_lookup_dictionaries(self):
        """Loads the embeddings and the lookup dictionaries."""
        if os.path.exists(self.embeddings_path):
            try:
                self.entity_emb = np.load(os.path.join(self.embeddings_path, "entity_embeds.npy"))
                
                self.relation_emb = np.load(os.path.join(self.embeddings_path, "relation_embeds.npy"))

                # load the dictionaries
                with open(os.path.join(self.embeddings_path, "entity_ids.del")) as f:
                    self.ent2id = {rdflib.URIRef(ent): int(idx) for idx, ent in csv.reader(f, delimiter='\t')}
                    self.id2ent = {v: k for k, v in self.ent2id.items()}
                with open(os.path.join(self.embeddings_path, "relation_ids.del")) as f:
                    self.rel2id = {rdflib.URIRef(rel): int(idx) for idx, rel in csv.reader(f, delimiter='\t')}
                    self.id2rel = {v: k for k, v in self.rel2id.items()}
                print(f"[Embeddings] Loaded {len(self.ent2id)} entities and {len(self.rel2id)} relations from embeddings.")

                print("[Embeddings] Loading entity labels from the knowledge graph...")
                self.ent2lbl = {ent: str(lbl) for ent, lbl in self.kg_handler.graph.subject_objects(RDFS.label)}
                self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}
                print(f"[Embeddings] Loaded {len(self.ent2lbl)} entity2label entries and {len(self.lbl2ent)} label2entity entries")

                self.all_entity_labels = list(self.ent2lbl.values())
                print(f"[Embeddings] Loaded {len(self.all_entity_labels)} entity labels (all)")
            except Exception as e:
                print(f"[Embeddings] Error loading embeddings: {e}")
                raise e

        else:
            raise FileNotFoundError(f"[ERROR] Embeddings directory not found: {self.embeddings_path}. Please run the embedding script to generate the embeddings.")

    def _sanitize_question(self, question: str) -> str:
        """Sanitizes the question."""
        # Strip outside whitespace
        question = question.strip()

        # Remove prefixes
        question = re.sub(r"please answer this question with a factual approach:\s*", "", question, flags=re.IGNORECASE)
        question = re.sub(r"please answer this question with an embedding approach:\s*", "", question, flags=re.IGNORECASE)
        question = re.sub(r"please answer this question:\s*", "", question, flags=re.IGNORECASE)
        
        print(f"[Sanitized Question] {question}")

        return question

    def answer(self, question: str, submode: str = "factual") -> str:
        """Pipeline for answering a natural language question based on different approaches."""
        # Set the object wide submode (factual or embedding)
        self.submode = submode
        
        # Sanitize the question
        question = self._sanitize_question(question)
        
        # Extract entities
        best_entity_uri, best_entity_label, property_uri, property_label = self.extract_entities_and_properties(question)

        # Check if entity and property were found
        if not best_entity_uri or not property_uri or not best_entity_label:
            return "I'm sorry, I couldn't find any informations in my knowledge graph that would make it possible to answer your question :("

        if self.submode == "factual":
            print(f"[QA Handler] Answering question with factual approach...")
            return self._answer_factual(question, best_entity_uri, property_uri, best_entity_label)
        elif self.submode == "embedding":
            print(f"[QA Handler] Answering question with embedding approach...")
            return self._answer_embedding(question, best_entity_uri, property_uri, best_entity_label)
        else:
            raise ValueError(f"Invalid submode: {self.submode}")

    def extract_entities_and_properties(self, question: str):
        """Extracts entities and properties from the question."""
        # Use EntityExtractor for entity extraction
        _potential_entities = self.entity_extractor.extract_entities(question, use_fuzzy_match=False)
        
        # Will store the entity property candidates from ner or be empty if the fast approach fails -> fallback to brute force
        entity_property_candidates = []
        NUM_OF_POTENTIAL_ENTITIES_TO_CONSIDER = 2  # Number of potential entities to consider for the fast approach (helpful if first extracted potential entity is not the correct one)
        if _potential_entities:
            for _potential_entity in _potential_entities[:NUM_OF_POTENTIAL_ENTITIES_TO_CONSIDER]:  # Loop through the first 2 entities
                # Find entites by label - prioritize difficult entities
                top_k_entities = self.entity_extractor.find_entities_by_label(_potential_entity['text'])
                
                if top_k_entities:
                    # Identify the property for EACH entity
                    for entity_uri, entity_label in top_k_entities:
                        property_uri, property_label = self.identify_property_for_entity(question, entity_uri, fuzzy_threshold=80)
                        if property_uri:
                            entity_property_candidates.append((entity_uri, entity_label, property_uri, property_label))
        
        # Identify the best candidate
        if not entity_property_candidates:  # This runs if the fast approach fails
            brute_force_entity_property_candidates = []
            # Brute force fallback as a last resort
            print(f"[Final Entity Selection] No candidates found, brute force fallback...")
            # TODO: Inform user that this might take a while...
            # Use EntityExtractor's brute force method
            brute_force_entities = self.entity_extractor.brute_force_extract_entities(question)
            if not brute_force_entities:
                # TODO: Embedding fallback?
                return None, None, None, None
            main_entity = brute_force_entities[0]  # assuming the longest match is the main entity
            all_entity_uris = self._find_ambiguous_entity_uris(main_entity)  # Find all entity URIs for the main entity label
            if not all_entity_uris:
                # TODO: Embedding fallback?
                return None, None, None, None
            for entity_uri in all_entity_uris:
                property_uri, property_label = self.identify_property_for_entity(question, entity_uri, fuzzy_threshold=80)
                if property_uri:
                    brute_force_entity_property_candidates.append((entity_uri, main_entity, property_uri, property_label))
            if not brute_force_entity_property_candidates:
                # TODO: Embedding fallback?
                return None, None, None, None
            else:
                print(f"[Final Entity Selection] Brute force candidates found: {brute_force_entity_property_candidates}")
                if len(brute_force_entity_property_candidates) == 1:
                    best_entity_uri, best_entity_label, property_uri, property_label = brute_force_entity_property_candidates[0]
                    print(f"[Final Entity Selection] Single candidate: {best_entity_label} --> {best_entity_uri}, property: {property_label}")
                else:
                    # Select best entity looking at similarity between question (with replaced property synonyms) and entity label
                    # TODO: Maybe filter out low score best entities
                    best_entity_uri, best_entity_label, property_uri, property_label, score = self.select_best_entity(self._replace_synonyms_in_question(question), brute_force_entity_property_candidates)
                    print(f"[Final Entity Selection] Best match: {best_entity_label} --> {best_entity_uri}, property: {property_label} (score={score:.3f})")

        elif len(entity_property_candidates) == 1:
            best_entity_uri, best_entity_label, property_uri, property_label = entity_property_candidates[0]
            print(f"[Final Entity Selection] Single candidate: {best_entity_label} --> {best_entity_uri}, property: {property_label}")
        elif len(entity_property_candidates) > 1:
            # Select best entity looking at similarity between question (with replaced property synonyms) and entity label
            # TODO: Maybe filter out low score best entities
            best_entity_uri, best_entity_label, property_uri, property_label, score = self.select_best_entity(self._replace_synonyms_in_question(question), entity_property_candidates)
            print(f"[Final Entity Selection] Best match: {best_entity_label} --> {best_entity_uri}, property: {property_label} (score={score:.3f})")
        
        return best_entity_uri, best_entity_label, property_uri, property_label

    def _answer_embedding(self, question, best_entity_uri, property_uri, best_entity_label) -> str:
        """Answers a question using the embedding model."""
        # Search for the most likely answer entity using embedding similarity based on extracted entity and property
        ent_emb = self.entity_emb[self.ent2id[best_entity_uri]]
        property_uri = WDT[property_uri.split("/")[-1]]
        rel_emb = self.relation_emb[self.rel2id[property_uri]]
        lhs = ent_emb + rel_emb
        distances = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        most_likely = distances.argsort()[0]
        result = {}
        result["answerItem"] = self.id2ent[most_likely]
        
        final_entity_metadata = self.kg_handler.get_entity_metadata_local(best_entity_uri)
        final_entity_metadata["entity_label"] = best_entity_label

        # Format the final answer using an LLM with a prompt for natural language formatting from the prompt manager
        return self.format_answer([result], question, entity_metadata=final_entity_metadata)

    def _answer_factual(self, question, best_entity_uri, property_uri, best_entity_label) -> str:
        """Answers a factual question."""
        # Execute the SPARQL query based on extracted entity and property
        results = self.execute_entity_property_query(best_entity_uri, property_uri)
        
        final_entity_metadata = self.kg_handler.get_entity_metadata_local(best_entity_uri)
        final_entity_metadata["entity_label"] = best_entity_label

        # Format the final answer using an LLM with a prompt for natural language formatting from the prompt manager
        return self.format_answer(results, question, entity_metadata=final_entity_metadata)

    def _compute_score(self, question: str, property: str, metadata: Dict[str, List[str]]) -> float:
        """Compute compatibility between question and entity metadata."""
        score = 0.0
        
        # Semantic similarity
        q_emb = self.embedding_handler.model.encode(question, convert_to_tensor=True)
        meta_text = " ".join(metadata["types"]) + " " + metadata["description"]
        m_emb = self.embedding_handler.model.encode(meta_text, convert_to_tensor=True)
        score += float(util.cos_sim(q_emb, m_emb))

        # Domain matching boost
        domain_expectations = {
            "director": ["film", "movie", "television", "episode"],
            "genre": ["film", "book", "music", "novel", "song", "album"],
            "publication date": ["film", "song", "book", "album"],
            "author": ["book", "novel"],
            "composer": ["film", "song", "album"],
            "cast member": ["film", "tv", "movie"],
            "performer": ["film", "tv", "movie"],
            "founded by": ["company", "organization", "startup"],
            "location": ["place", "city", "country"],
            "award received": ["film", "person", "book"],
        }
        expected = domain_expectations.get(property, [])
        if any(word in metadata["description"] or word in " ".join(metadata["types"]) for word in expected):
            score += 0.5
        return score

    def select_best_entity(self, question: str, candidates: List[Tuple[str, str, str, str]]) -> Tuple[str, str, str, str, float]:
        """Selects the best entity from the candidates based on the question and property.
        Returns the best entity URI, label, property URI, property label, and score: (entity_uri, entity_label, property_uri, property_label, score)"""
        scores = []

        for entity_uri, entity_label, property_uri, property_label in candidates:
            metadata = self.kg_handler.get_entity_metadata_local(entity_uri)
            print(f"[Best Entity Selection] Metadata for {entity_uri}: {metadata}")
            s = self._compute_score(question, property_label, metadata)
            scores.append((entity_uri, entity_label, property_uri, property_label, s))

        best = max(scores, key=lambda x: x[4])
        return best
    
    def _find_ambiguous_entity_uris(self, entity: str) -> List[str]:
        """Finds all URIs for the given entity by querying the knowledge graph.
        Returns a list of entity URIs found in the knowledge graph."""
        entity = entity.strip()
        
        # Generate all possible entity variations to handle different dash types
        entity_variations = self._generate_entity_variations(entity)
        
        all_uris = set()  # Use set to avoid duplicates
        
        # Try each variation and collect all results
        for variation in entity_variations:
            query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT DISTINCT ?entity WHERE {{
                    ?entity rdfs:label "{variation}" .
                    FILTER(LANG("{variation}") = '' || LANGMATCHES(LANG("{variation}"), 'en'))
                }}
            """
            
            results = self.kg_handler.sparql_query(query)
            
            if isinstance(results, list) and results:
                uris = [row['entity'] for row in results if 'entity' in row]
                all_uris.update(uris)  # Add all URIs to the set
            elif isinstance(results, dict) and 'error' in results:
                print(f"[Entity URI Search] SPARQL query error for entity '{entity}': {results['error']}")
        
        return list(all_uris)  # Convert set back to list
    
    def _generate_entity_variations(self, entity: str) -> List[str]:
        """Generate all possible variations of an entity name to handle different dash types.
        Returns a list of entity variations.
        """
        variations = [entity]  # Start with the original
        
        # Add variations with different dash types
        if '-' in entity:
            variations.append(entity.replace('-', '–'))  # Replace hyphen with em-dash
        if '–' in entity:
            variations.append(entity.replace('–', '-'))  # Replace em-dash with hyphen
            
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                unique_variations.append(variation)
                seen.add(variation)
                
        return unique_variations

    def _replace_synonyms_in_question(self, question: str) -> str:
        """Replaces synonyms in the question with their canonical property names."""
        # First, collect all synonyms and sort by length
        all_synonyms = []
        for prop_name, synonyms_list in self.property_synonym_map.items():
            for syn in synonyms_list:
                all_synonyms.append((syn, prop_name))
        
        # Sort by synonym length (longest first) to avoid partial matches
        all_synonyms.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Use word boundaries to ensure we only replace whole words/phrases
        for syn, prop_name in all_synonyms:
            # Use word boundaries for single words, or exact phrase matching for multi-word phrases
            if ' ' in syn:
                # Multi-word phrase - use exact phrase matching
                pattern = re.compile(re.escape(syn), re.IGNORECASE)
                if pattern.search(question):
                    print(f"[Property Matching] Found multi-word synonym '{syn}' -> '{prop_name}'")
                    question = pattern.sub(prop_name, question, count=1)
                    break
            else:
                # Single word - use word boundaries
                pattern = re.compile(r'\b' + re.escape(syn) + r'\b', re.IGNORECASE)
                if pattern.search(question):
                    print(f"[Property Matching] Found single-word synonym '{syn}' -> '{prop_name}'")
                    question = pattern.sub(prop_name, question, count=1)
                    break
        return question

    def identify_property_for_entity(self, question: str, entity_uri: str, fuzzy_threshold: int = 80) -> Tuple[Optional[str], Optional[str]]:
        """Identifies the property the user is asking about based on a main entity URI.
        Returns the property URI and the property label if found, (None, None) otherwise: (property_uri, property_label)"""
        # 1. Replace synonyms in the question with their canonical property names
        processed_question = self._replace_synonyms_in_question(question.lower())  # properties are always lowercase

        print(f"[Property Matching] Processed question (with synonyms replaced): {processed_question}")
        
        # 2. Entity-specific property filtering
        if entity_uri:
            print(f"[Property Matching] Using entity-specific filtering for: {entity_uri}")
            entity_props = self.kg_handler.get_entity_property_labels(entity_uri)
            if entity_props:
                entity_prop_labels = {p.get('label', '').lower(): p.get('property') 
                                    for p in entity_props if p.get('label')}
            else:
                # TODO: This could mean that the entity has no properties we should distinguish between no properties and an error
                print(f"[Property Matching] Sparql query returned an empty list for entity '{entity_uri}'")
                return None, None
            
            # Match entity-specific properties to the processed question
            match = process.extractOne(
                processed_question,
                entity_prop_labels.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=fuzzy_threshold  # Lower threshold for entity-specific
            )
            
            if match and match[1] >= fuzzy_threshold:
                print(f"[Property Matching] Entity-specific: '{match[0]}' (score: {match[1]}, threshold: {fuzzy_threshold})")
                # Return the property URI and the property label
                return entity_prop_labels[match[0]], match[0]
            else:
                print(f"[Property Matching] No property found for entity '{entity_uri}' in question '{question}' with threshold {fuzzy_threshold}")
                return None, None

    def execute_entity_property_query(self, entity_uri: str, property_uri: str) -> Dict[str, Any]:
        """Executes the query for the given entity and property.
        Returns the results if found, None otherwise."""
        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?answerLabel ?answerItem WHERE {{
                <{entity_uri}> <{property_uri}> ?answerItem .
            }}
        """
        results = self.kg_handler.sparql_query(query)
        if isinstance(results, list) and results:
            return results
        elif isinstance(results, dict) and 'error' in results:
            print(f"[Entity Property Query] SPARQL query error for entity '{entity_uri}' and property '{property_uri}': {results['error']}")
            return None
        else:
            print(f"[Entity Property Query] Unknown error: {results}")
            return None

    def format_answer(self, results: List[Dict[str, Any]], question: str, entity_metadata: Dict) -> str:
        """Formats the answer for the user using LLM with prompt for natural language formatting."""
        if isinstance(results, dict) and "error" in results:
            print(f"ERROR: {results['error']}")
            return "I could not find an answer to your question in my knowledge graph.. I'm sorry about that."

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
                    label = self.kg_handler.get_label_for_uri(answer_item)
                    if label != answer_item:  # If we got a proper label
                        answers.append(label)
                    else:
                        answers.append(answer_item)
                else:
                    # It's a literal value (like a date)
                    answers.append(str(answer_item))
        
        if not answers:
            return "I could not find an answer to your question in my knowledge graph.. I'm sorry about that."

        # Remove duplicates while preserving order
        unique_answers = []
        seen = set()
        for answer in answers:
            if answer not in seen:
                unique_answers.append(answer)
                seen.add(answer)

        # Use LLM to format the answer naturally
        try:
            # Prepare the raw data for the LLM
            raw_data = {
                "answers": unique_answers,
                "answers_count": len(unique_answers),
                "question_entity_metadata": entity_metadata
            }

            # print(f"[Answer Formatting] Raw data: \n```json\n{json.dumps(raw_data, indent=2, ensure_ascii=False)}\n```")
            
            # Get the prompt for natural language formatting
            prompt = self.prompt_manager.get_prompt(
                "qa_formatter",
                question=question,
                raw_data=json.dumps(raw_data, indent=2, ensure_ascii=False)
            )
            
            # Generate a natural language response using the LLM
            response = self.llm_handler.generate_response(prompt)
            
            if response['success']:
                if self.submode == "factual":
                    return "Based on my knowledge graph: " + response['content']
                elif self.submode == "embedding":
                    # Get type id from answer
                    if len(unique_answers) == 1:
                        answer = unique_answers[0]
                        P31 = WDT.P31
                        # Get URI from label using lbl2ent mapping
                        answer_uri = self.lbl2ent.get(answer)
                        if answer_uri:
                            answer_uri_ref = rdflib.URIRef(answer_uri)
                            type_objects = list(self.kg_handler.graph.objects(answer_uri_ref, P31))
                            if type_objects:
                                # Get the first type entity URI
                                type_uri = str(type_objects[0])
                                # Extract Q-id from type_uri (e.g., "http://www.wikidata.org/entity/Q201658" -> "Q201658")
                                q_id = type_uri.split("/")[-1] if "/" in type_uri else ""
                                type_string = " (type: " + q_id + ")" if q_id else ""
                            else:
                                type_string = ""
                            return "Based on my embeddings: " + response['content'] + type_string
                    else:
                        return "Based on my embeddings: " + response['content']
                else:
                    return response["content"]
            else:
                # Fallback to simple formatting if LLM fails
                return "Based on my knowledge graph: " + ", ".join(unique_answers)
                
        except Exception as e:
            print(f"Error formatting answer with LLM: {e}")
            # Fallback to simple formatting if LLM fails
            return "Based on my knowledge graph: " + ", ".join(unique_answers)
