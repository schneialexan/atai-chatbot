# app/multimedia_handler.py
import os
import json
import re
import random
from typing import Dict, List, Optional, Tuple
import rdflib
from rdflib.namespace import RDFS
import spacy

from app.kg_handler import LocalKnowledgeGraph
from app.entity_extractor import EntityExtractor

# define some prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')

FILM_CLASSES = {
    WD.Q11424, # film
    WD.Q202866,# animated film
    WD.Q24856, # film series
}

PERSON_CLASSES = {
    WD.Q5,  # human
}


class MultimediaHandler:
    def __init__(self, kg_handler: LocalKnowledgeGraph, images_json_path: str = "dataset/additional/images.json"):
        """
        Initialize the MultimediaHandler.
        
        Args:
            kg_handler: LocalKnowledgeGraph instance for entity lookups
            images_json_path: Path to the images.json file
        """
        self.kg_handler = kg_handler
        self.images_json_path = images_json_path
        self.images_data = self._load_images_data()
        
        # Load entity labels from knowledge graph (for EntityExtractor)
        print("[Multimedia Handler] Loading entity labels from knowledge graph...")
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.kg_handler.graph.subject_objects(RDFS.label)}
        self.all_entity_labels = list(self.ent2lbl.values())
        print(f"[Multimedia Handler] Loaded {len(self.ent2lbl)} entity labels")
        
        # Initialize spaCy NER model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize EntityExtractor (no property names needed for multimedia)
        self.entity_extractor = EntityExtractor(
            kg_handler=self.kg_handler,
            nlp=self.nlp,
            property_names=[],  # No property filtering needed for multimedia
            ent2lbl=self.ent2lbl,
            all_entity_labels=self.all_entity_labels
        )

    def _load_images_data(self) -> List[Dict]:
        """
        Load images.json data.
        
        Returns:
            List of image entries from images.json
        """
        if not os.path.exists(self.images_json_path):
            print(f"ERROR: images.json not found at {self.images_json_path}")
            return []
        
        print(f"[Multimedia Handler] Loading images.json from {self.images_json_path}...")
        try:
            with open(self.images_json_path, "r", encoding="utf-8") as f:
                images_data = json.load(f)
            print(f"[Multimedia Handler] Loaded {len(images_data)} image entries")
            return images_data
        except Exception as e:
            print(f"ERROR: loading images.json: {e}")
            return []

    def _get_imdb_id_from_entity(self, entity_uri: str) -> Optional[str]:
        """
        Get IMDb ID for an entity from the knowledge graph.
        
        Args:
            entity_uri: Entity URI (string)
            
        Returns:
            IMDb ID (string) if found, None otherwise
        """
        uri_ref = rdflib.URIRef(entity_uri) if isinstance(entity_uri, str) else entity_uri
        imdb_values = list(self.kg_handler.graph.objects(uri_ref, WDT.P345))
        
        if not imdb_values:
            return None
        
        # Get the first IMDb ID and clean it
        imdb_value = imdb_values[0]
        imdb_str = str(imdb_value).strip()
        
        # Extract IMDb ID (handle cases where it might be in a URL or have prefixes)
        # Try exact match first (should be nm... or tt...)
        if imdb_str.startswith("nm") or imdb_str.startswith("tt"):
            return imdb_str
        
        # Try to extract nm... or tt... pattern
        match = re.search(r'(nm\d+|tt\d+)', imdb_str)
        if match:
            return match.group(1)
        
        return None

    def _search_images_by_imdb_id(self, imdb_id: str, is_film: bool) -> List[str]:
        """
        Search images.json for entries matching the IMDb ID.
        
        Args:
            imdb_id: IMDb ID to search for
            is_film: True if searching for film images, False for person images
            
        Returns:
            List of image paths matching the IMDb ID
        """
        matching_images = []
        
        for entry in self.images_data:
            img_path = entry.get("img", "")
            if not img_path:
                continue
            
            cast_ids = entry.get("cast", []) or []
            movie_ids = entry.get("movie", []) or []
            
            if is_film:
                # For films, check if IMDb ID is in the movie list
                if imdb_id in movie_ids:
                    matching_images.append(img_path)
            else:
                # For persons, check if IMDb ID is in the cast list and the entry is a profile
                if imdb_id in cast_ids and entry.get("type", "") == "profile":
                    matching_images.append(img_path)
        print(f"[Multimedia Handler] Found {len(matching_images)} images for {imdb_id}")
        return matching_images

    def is_film(self, uri) -> bool:
        """
        Check if entity URI is a film.
        
        Args:
            uri: Entity URI (URIRef or string)
            
        Returns:
            True if entity is a film, False otherwise
        """
        uri_ref = rdflib.URIRef(uri) if isinstance(uri, str) else uri
        types = set(self.kg_handler.graph.objects(uri_ref, WDT.P31))
        return bool(types & FILM_CLASSES)
    
    def is_person(self, uri) -> bool:
        """
        Check if entity URI is a person.
        
        Args:
            uri: Entity URI (URIRef or string)
            
        Returns:
            True if entity is a person, False otherwise
        """
        uri_ref = rdflib.URIRef(uri) if isinstance(uri, str) else uri
        types = set(self.kg_handler.graph.objects(uri_ref, WDT.P31))
        return bool(types & PERSON_CLASSES)
    
    def _extract_entity_from_question(self, question: str) -> List[Tuple[str, bool]]:
        """
        Extract entity URI from question using EntityExtractor, filtered to only films or persons.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (entity_uri, is_film) if found, None otherwise
        """
        # Use EntityExtractor with fuzzy matching enabled
        potential_entities = self.entity_extractor.extract_entities(
            question, 
            use_fuzzy_match=True
        )
        
        candidate_uris = []
        
        if potential_entities:
            # Try to find entity URIs for the extracted entities
            for entity_dict in potential_entities:
                entity_text = entity_dict.get('text', '')
                if entity_text:
                    # Use find_entities_by_label to get entity URIs
                    top_k_entities = self.entity_extractor.find_entities_by_label(entity_text)
                    for entity_uri, entity_label in top_k_entities:
                        candidate_uris.append(str(entity_uri))
        
        # Fallback to brute force extraction
        if not candidate_uris:
            print("[Multimedia Handler] Using brute force entity extraction...")
            brute_force_entities = self.entity_extractor.brute_force_extract_entities(question)
            if brute_force_entities:
                # Find URIs for brute force entities
                for entity_label in brute_force_entities:
                    top_k_entities = self.entity_extractor.find_entities_by_label(entity_label)
                    for entity_uri, _ in top_k_entities:
                        candidate_uris.append(str(entity_uri))
        
        if not candidate_uris:
            return []

        entity_results = []
        # Filter to only films or persons using is_film and is_person
        for uri in candidate_uris:
            if self.is_film(uri):
                entity_label = self.kg_handler.get_label_for_uri(uri)
                print(f"[Multimedia Handler] Extracted film entity: {entity_label} ({uri}) from candidates")
                entity_results.append((uri, True))
            elif self.is_person(uri):
                entity_label = self.kg_handler.get_label_for_uri(uri)
                print(f"[Multimedia Handler] Extracted person entity: {entity_label} ({uri}) from candidates")
                entity_results.append((uri, False))
        
        if not entity_results:
            print(f"[Multimedia Handler] No films or persons found among {len(candidate_uris)} candidate entities")
        else:
            print(f"[Multimedia Handler] Found {len(entity_results)} entities that are films or persons from {len(candidate_uris)} candidate entities")
            return entity_results

    def _sanitize_question(self, question: str) -> str:
        """Sanitize the question by removing multimedia-related phrases."""
        question = question.strip()
        
        print(f"[Multimedia Handler] Sanitized question: {question}")
        return question

    def get_image(self, query: str, return_random_image: bool = False) -> str:
        """
        Returns an image path based on query (e.g. 'Show me a picture of Halle Berry').
        
        Args:
            query: User's question about an entity
            
        Returns:
            Image path in format "image:{path}" or error message
        """
        # Sanitize question
        question = self._sanitize_question(query)
        
        # Extract entity URI from question (filtered to only films or persons)
        entity_results = self._extract_entity_from_question(question)
        if not entity_results:
            return "I'm sorry, I couldn't identify the person or movie you're asking about. Please try rephrasing your question."
        
        # Loop through entity results until a valid image is found
        for entity_uri, is_film in entity_results:
            # Get IMDb ID from knowledge graph
            imdb_id = self._get_imdb_id_from_entity(entity_uri)
            if not imdb_id:
                print(f"[Multimedia Handler] No IMDb ID found for {entity_uri}, trying next entity...")
                continue
            
            print(f"[Multimedia Handler] Found IMDb ID: {imdb_id} (type: {'film' if is_film else 'person'})")
            
            # Search images.json for matching entries
            matching_images = self._search_images_by_imdb_id(imdb_id, is_film)
            
            if not matching_images:
                print(f"[Multimedia Handler] No images found for {entity_uri}, trying next entity...")
                continue
            
            # Found a valid image, return it
            image_path = random.choice(matching_images) if return_random_image else matching_images[0]
            result = f"image:{image_path}"
            
            print(f"[Multimedia Handler] Returning image https://files.ifi.uzh.ch/ddis/teaching/2025/ATAI/dataset/images/{image_path}")
            return result
        
        # If we get here, no valid image was found for any entity
        return "I'm sorry, I couldn't find any images for the person or movie you're asking about. Please try a different query."
