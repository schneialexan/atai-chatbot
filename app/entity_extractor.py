import os
import re
import json
import difflib
from typing import List, Dict, Optional, Tuple

import spacy
from rapidfuzz import process, fuzz

from app.kg_handler import LocalKnowledgeGraph

CACHED_ENTITIES_FILE = "app/cached_entities.json"


class EntityExtractor:
    """Sophisticated entity extraction module with multiple fallback strategies.
    
    Supports:
    - Pattern-based extraction for complex entities (quoted titles, colons, special characters)
    - spaCy-based named entity recognition
    - Fuzzy matching for misspelled or variant entities
    - Brute force fallback when initial extraction fails
    - Fuzzy matching with longest-match strategy (optional, requires embeddings)
    """
    
    def __init__(self, kg_handler: LocalKnowledgeGraph, nlp, property_names: List[str],
                 ent2lbl, all_entity_labels):
        """
        Initialize the EntityExtractor.
        
        Args:
            kg_handler: LocalKnowledgeGraph instance for entity lookups
            nlp: spaCy language model for NER
            property_names: List of property names to filter out from entity matches
            ent2lbl: dict mapping entity URIs to labels (for find_entities_by_label)
            all_entity_labels: list of all entity labels (for find_entities_by_label)
        """
        self.kg_handler = kg_handler
        self.nlp = nlp
        self.property_names = property_names
        self.ent2lbl = ent2lbl
        self.all_entity_labels = all_entity_labels
        
        # Load cached entities internally
        self.entities = self._load_cached_entities()

        print(f"[Entity Extractor] Initialized with {len(self.entities)} cached entity labels")
    
    def _load_cached_entities(self) -> List[str]:
        """Gets all entity labels from the knowledge graph, with caching.
        Returns a list of entity labels for efficient entity extraction."""
        if os.path.exists(CACHED_ENTITIES_FILE):
            with open(CACHED_ENTITIES_FILE, "r", encoding="utf-8") as f:
                print(f"Loading cached entity labels from {CACHED_ENTITIES_FILE}")
                return json.load(f)

        print("Querying knowledge graph for all entity labels...")
        entity_labels = self.kg_handler.get_all_entities()
        print(f"Found {len(entity_labels)} unique entity labels")
        
        with open(CACHED_ENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(entity_labels, f, indent=1, ensure_ascii=False)
        print(f"Cached entity labels saved to {CACHED_ENTITIES_FILE}")
        return entity_labels
    
    def extract_entities(self, question: str, use_fuzzy_match: bool = False, 
                        fuzzy_cutoff: float = 0.7, max_fuzzy_entities: int = 3) -> List[Dict[str, str]]:
        """
        Unified entity extraction method that combines all strategies.
        
        Args:
            question: User's question/query
            use_fuzzy_match: Whether to use fuzzy matching for misspelled or variant entities
            fuzzy_cutoff: Similarity threshold for fuzzy matching (0.0 to 1.0, higher = more strict)
            
        Returns:
            List of entity dictionaries with 'text' and 'label' keys
        """
        potential_entities = []
        
        # 1. Difficult entities extraction (pattern-based)
        difficult_entities = self.extract_difficult_entities(question)
        for entity in difficult_entities:
            potential_entities.append({"text": entity, "label": "WORK_OF_ART"})
        
        # 2. NER extraction
        ner_entities = self.ner_entity_extraction(question)
        if ner_entities:
            # Only add NER entities if they don't overlap with difficult entities
            for ner_ent in ner_entities:
                if not any(ner_ent['text'] in entity for entity in difficult_entities):
                    potential_entities.append(ner_ent)
        
        # 3. Fuzzy matching for misspelled or variant entities if enabled
        if use_fuzzy_match:
            fuzzy_entities = self.fuzzy_match_extract_entities(question, cutoff=fuzzy_cutoff)
            cnt = 0
            for entity in fuzzy_entities:
                if cnt >= max_fuzzy_entities:
                    break
                cnt += 1
                # Check if entity already exists in potential_entities (case-insensitive)
                if not any(existing_ent['text'].lower() == entity.lower() for existing_ent in potential_entities):
                    potential_entities.append({"text": entity, "label": "ENTITY"})
        
        return potential_entities
    
    def ner_entity_extraction(self, question: str) -> List[Dict[str, str]]:
        """Extracts entities from the question using NER."""
        doc = self.nlp(question)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        print(f"[NER] Entities detected: {entities}")
        return entities

    def extract_difficult_entities(self, question: str) -> List[str]:
        """Extract difficult entities using pattern matching."""
        entities = []
        
        # Common patterns based on cached entities - targeting difficult edge cases
        patterns = [
            # Entities wrapped in quotes (highest priority)
            r"'([^']+)'",  # Single quotes
            r'"([^"]+)"',  # Double quotes
            r'[\u2018]([^\u2019]+)[\u2019]',  # Single curly quotes
            r'[\u201C]([^\u201D]+)[\u201D]',  # Double curly quotes
            
            # Star Wars patterns with dashes and colons
            r'Star Wars: Episode [IVX]+[^?]*',
            r'Star Wars Episode [IVX]+[^?]*',
            
            # X-Men patterns with colons and dashes
            r'X-Men Beginnings[^?]*',
            r'X-Men: [A-Z][a-z]+[^?]*',
            r'X-Men Origins: [A-Z][a-z]+[^?]*',
            
            # Complex titles with colons and em-dashes
            r'The Hobbit: [A-Z][a-z]+[^?]*',
            r'Mission: Impossible[^?]*',
            r'Captain America: [A-Z][a-z]+[^?]*',
            r'John Wick: [A-Z][a-z]+[^?]*',
            r'Sweeney Todd: [A-Z][a-z]+[^?]*',
            
            # Titles with multiple special characters
            r'[A-Z][a-z]+: [A-Z][a-z]+ – [A-Z][a-z]+[^?]*',
            r'[A-Z][a-z]+ [A-Z][a-z]+: [A-Z][a-z]+ – [A-Z][a-z]+[^?]*',
            
            # Specific difficult cases from cache
            r'The Chronicles of Narnia: [A-Z][a-z]+[^?]*',
            r'The Lord of the Rings: [A-Z][a-z]+[^?]*',
            r'The X-Files: [A-Z][a-z]+[^?]*',
            r'Night at the Museum: [A-Z][a-z]+[^?]*',
            r'Ice Age: [A-Z][a-z]+[^?]*',
            r'Hellraiser [IVX]+: [A-Z][a-z]+[^?]*',
            
            # Movies with numbers and special characters
            r'[0-9]+: [A-Z][a-z]+[^?]*',
            r'[A-Z][a-z]+ [0-9]+: [A-Z][a-z]+[^?]*',
            
            # Complex titles with multiple colons
            r'[A-Z][a-z]+: [A-Z][a-z]+: [A-Z][a-z]+[^?]*'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                entity = match.strip()
                # Check if entity already exists (case-insensitive)
                if entity and not any(existing.lower() == entity.lower() for existing in entities):
                    entities.append(entity)
        
        print(f"[Difficult Entity Extraction] Found difficult entities: {entities}")
        return entities

    def fuzzy_match_extract_entities(self, question: str, cutoff: float = 0.7, max_matches_per_mention: int = 1) -> List[str]:
        """
        Extract entities using fuzzy matching to handle misspellings and variations.
        
        This method extracts potential entity mentions from the question and then uses
        rapidfuzz for fast fuzzy string matching to find the closest matching entities 
        in the cached entity list.
        
        Args:
            question: User's question/query
            cutoff: Similarity threshold for fuzzy matching (0.0 to 1.0, higher = more strict)
            max_matches_per_mention: Maximum number of matches to return per potential mention
            
        Returns:
            List of matched entity labels
        """
        print(f"[Fuzzy Matching] Processing question: '{question}'")
        
        # Extract potential entity mentions from the question
        potential_mentions = self._extract_potential_entity_mentions(question)
        
        if not potential_mentions:
            print("[Fuzzy Matching] No potential entity mentions found")
            return []
        
        print(f"[Fuzzy Matching] Found {len(potential_mentions)} potential entity mentions: {potential_mentions}")
        
        matched_entities = []
        cached_entity_labels = self.entities
        
        # Convert cutoff from 0-1 scale to 0-100 scale (rapidfuzz uses 0-100)
        score_cutoff = cutoff * 100
        
        # For each potential mention, find the closest matching entity
        for mention in potential_mentions:
            # Skip very short mentions (likely false positives)
            if len(mention.strip()) < 3:
                continue
            
            # Skip if mention is in property names
            if mention.lower() in [prop.lower() for prop in self.property_names]:
                continue
            
            # Use rapidfuzz.process.extract for efficient fuzzy matching
            # Returns list of tuples: (matched_string, score, index)
            matches = process.extract(
                mention,
                cached_entity_labels,
                limit=max_matches_per_mention,
                score_cutoff=score_cutoff,
                scorer=fuzz.WRatio  # Weighted ratio - good for handling different string lengths
            )
            
            if matches:
                for match, score, _ in matches:
                    # Additional validation: check if the match is reasonable
                    # Use token_sort_ratio for better handling of word order differences
                    token_similarity = fuzz.token_sort_ratio(mention.lower(), match.lower())
                    if token_similarity >= score_cutoff:
                        if match not in matched_entities:
                            matched_entities.append(match)
                            # Convert score back to 0-1 scale for display
                            similarity_ratio = score / 100.0
                            print(f"[Fuzzy Matching] Matched '{mention}' -> '{match}' (similarity: {similarity_ratio:.2f})")
        
        # Sort matches by length (longest first) and apply case matching safety check
        matched_entities = self._select_entities_by_longest_match(matched_entities, question)
        
        print(f"[Fuzzy Matching] Fuzzy matches (sorted by length): {matched_entities}")
        return matched_entities
    
    def _extract_potential_entity_mentions(self, question: str) -> List[str]:
        """
        Extract potential entity mentions from the question for fuzzy matching.
        Uses heuristics to identify likely entity names.
        
        Args:
            question: User's question/query
            
        Returns:
            List of potential entity mention strings
        """
        mentions = []
        
        # 1. Extract capitalized phrases (likely proper nouns)
        # Pattern: sequences of capitalized words
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized_matches = re.findall(capitalized_pattern, question)
        mentions.extend(capitalized_matches)
        
        # 2. Extract quoted strings (often entity names)
        quoted_patterns = [
            r"'([^']+)'",  # Single quotes
            r'"([^"]+)"',  # Double quotes
            r'[\u2018]([^\u2019]+)[\u2019]',  # Single curly quotes
            r'[\u201C]([^\u201D]+)[\u201D]',  # Double curly quotes
        ]
        for pattern in quoted_patterns:
            quoted_matches = re.findall(pattern, question)
            mentions.extend(quoted_matches)
        
        # 3. Extract phrases after common entity-indicating words
        entity_indicators = [
            r'(?:movie|film|show|series|actor|director|star|starring|directed by|called|named)\s+["\']?([A-Z][^?.!]+)',
            r'(?:the|a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        for pattern in entity_indicators:
            indicator_matches = re.findall(pattern, question, re.IGNORECASE)
            mentions.extend(indicator_matches)
        
        # Clean and deduplicate mentions
        cleaned_mentions = []
        for mention in mentions:
            mention = mention.strip()
            # Remove trailing punctuation
            mention = re.sub(r'[.,!?;:]+$', '', mention)
            # Skip very short or common words
            if len(mention) >= 3 and mention.lower() not in ['the', 'and', 'or', 'but', 'for', 'with']:
                if mention not in cleaned_mentions:
                    cleaned_mentions.append(mention)
        
        return cleaned_mentions

    def find_entities_by_label(self, label: str, top_k: int = 5, get_close_matches_n: int = 1, get_close_matches_cutoff: float = 0.6) -> List[Tuple[str, str]]:
        """
        Finds the closest matching entity label and returns the top_k matching entity URIs.
        Requires embeddings lookup dictionaries (ent2lbl, all_entity_labels).
        
        Returns:
            List of tuples: [(entity_uri, entity_label), ...]
        """        
        match = difflib.get_close_matches(label, self.all_entity_labels, n=get_close_matches_n, cutoff=get_close_matches_cutoff)
        if not match:
            return []
        # return the top_k matching entity URI(s)
        best_label = match[0]
        candidates = [(ent, self.ent2lbl[ent]) for ent, lbl in self.ent2lbl.items() if lbl == best_label]
        print(f"[Entity Label Matching] Found {len(candidates)} candidates for '{label}': {candidates[:top_k]}")
        return candidates[:top_k]

    def _find_entity_with_word_boundaries(self, question: str, entity_label: str) -> Optional[str]:
        """
        Find entity in question using flexible matching for complex entities.
        Supports multi-word entities with punctuation, colons, and special characters.
        
        Args:
            question: User's question (original case preserved)
            entity_label: Entity label to find
            
        Returns:
            Matched entity string if found, None otherwise
        """
        # For simple single-word entities: use \b boundaries
        if ' ' not in entity_label and not any(char in entity_label for char in [':', '–', '-', '(', ')']):
            pattern = re.compile(r'\b' + re.escape(entity_label) + r'\b', re.IGNORECASE)
            match = pattern.search(question)
            if match:
                return match.group()
        
        # For complex entities (with punctuation, colons, etc.): use flexible matching
        else:
            if entity_label in question:
                # Find the position and extract the original case version
                start_pos = question.find(entity_label)
                if start_pos != -1:
                    return question[start_pos:start_pos + len(entity_label)]
            
            # Second try: flexible regex pattern for punctuation variations
            # Create a flexible pattern that handles punctuation variations
            escaped_label = re.escape(entity_label)
            
            # Replace escaped punctuation with flexible patterns
            # Handle common punctuation variations
            flexible_pattern = escaped_label
            flexible_pattern = flexible_pattern.replace(r'\:', r'[:–-]?')  # Colon, em-dash, or hyphen
            flexible_pattern = flexible_pattern.replace(r'–', r'[–-]')  # Em-dash or hyphen
            flexible_pattern = flexible_pattern.replace(r'\-', r'[–-]')  # Hyphen variations
            
            # Use word boundaries for the start and end, but allow punctuation in between
            pattern = re.compile(r'\b' + flexible_pattern + r'\b', re.IGNORECASE)
            match = pattern.search(question)
            if match:
                return match.group()
        
        return None

    def _select_entities_by_longest_match(self, potential_entities: List[str], question: str) -> List[str]:
        """
        Select entities using longest match strategy - prioritize entities with longer matches.
        Also implements safety check for exact case matches.
        
        Args:
            potential_entities: List of entity label strings
            question: Original question for case matching
            
        Returns:
            List of selected entity labels (sorted by length, longest first)
        """
        if not potential_entities:
            print("[Entity Selection] No potential entities to select from")
            return []
        
        # Safety check: If we have both lowercase and proper case versions, prefer the one that exactly matches the question
        if len(potential_entities) == 2:
            entity1, entity2 = potential_entities[0], potential_entities[1]
            if entity1.lower() == entity2.lower() and entity1 != entity2:
                # Check which one appears exactly in the original question
                if entity1 in question and entity2 not in question:
                    print(f"[Entity Selection] Safety check: preferring exact case match '{entity1}' over '{entity2}'")
                    return [entity1]
                elif entity2 in question and entity1 not in question:
                    print(f"[Entity Selection] Safety check: preferring exact case match '{entity2}' over '{entity1}'")
                    return [entity2]
        
        # Sort entities by match length (longest first)        
        sorted_entities = sorted(potential_entities, key=len, reverse=True)
        return sorted_entities

    def brute_force_extract_entities(self, question: str) -> List[str]:
        """
        Last resort entity extraction method.
        
        This method is used as a last resort when the other entity extraction methods fail.
        It loops through all entity labels from the knowledge graph and tries to find an exact match in the question.
        
        Args:
            question: User's question/query
        Returns:
            List of matched entity labels
        """
        print(f"[Entity Extraction] Processing question: '{question}'")
        
        # Collect all potential entity matches
        potential_entities = []
        
        cached_entity_labels = self.entities
        sorted_entities = sorted(cached_entity_labels, key=len, reverse=True)

        # Exact word boundary matching (highest priority)
        for entity_label in sorted_entities:
            # Use original question for exact matching to preserve case
            match = self._find_entity_with_word_boundaries(question, entity_label)
            if match and match not in self.property_names:
                potential_entities.append(entity_label)
                print(f"[Entity Extraction] Exact match: '{entity_label}'")
        
        # Final: Select best entities using longest match strategy
        selected_entities = self._select_entities_by_longest_match(potential_entities, question)
        
        print(f"[Entity Extraction] Final selection: {selected_entities}")
        return selected_entities

