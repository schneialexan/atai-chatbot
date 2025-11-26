import rdflib
import logging
import spacy
import difflib
import os
import csv
import json
import re
import hashlib
import pickle
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rdflib.namespace import RDFS
from collections import defaultdict

from app.kg_handler import LocalKnowledgeGraph
from app.entity_extractor import EntityExtractor

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, use a simple pass-through function
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

logger = logging.getLogger(__name__)

WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')

ITEM_RATINGS_PATH = "dataset/ratings/item_ratings.csv"
USER_RATINGS_PATH = "dataset/ratings/user_ratings.csv"
LABEL_TO_URI_PATH = "app/label_to_uri.json"

FILM_CLASSES = {
    WD.Q11424, # film
    WD.Q24866, # television film
    WD.Q202866,# animated film
    WD.Q24856, # short film
}

# RELATIONAL props to follow (relational navigation)
RELATIONAL_PROPS = [
    WDT.P31,   # instance of
    WDT.P279,  # subclass of
    WDT.P144,  # based on
    WDT.P179,  # part of series
    WDT.P495,  # country of origin
]

# INTERESTING props to collect
INTERESTING_PROPS = [
    # Creative roles
    WDT.P170,  # creator
    WDT.P50,   # author
    WDT.P57,   # director
    WDT.P58,   # screenwriter
    WDT.P161,  # cast member
    WDT.P86,   # composer
    WDT.P272,  # production company
    WDT.P750,  # distributor
    WDT.P364,  # original language of work
    WDT.P495,  # country of origin
    
    # Thematic
    WDT.P136,  # genre
    WDT.P921,  # main subject
]

class MovieRecommender:
    def __init__(self, kg_handler: LocalKnowledgeGraph,
                 embeddings_path: str = "dataset/store/embeddings",
                 llm_handler=None, prompt_manager=None,
                 user_ratings_path: str = USER_RATINGS_PATH,
                 item_ratings_path: str = ITEM_RATINGS_PATH,
                 cache_dir: str = "dataset/store"):
        self.kg_handler = kg_handler
        self.embeddings_path = embeddings_path
        self.llm_handler = llm_handler
        self.prompt_manager = prompt_manager
        self.user_ratings_path = user_ratings_path
        self.item_ratings_path = item_ratings_path
        self.cache_dir = cache_dir

        self.nlp = spacy.load("en_core_web_trf")

        self.item_ratings = pd.read_csv(self.item_ratings_path)
        self.user_ratings = pd.read_csv(self.user_ratings_path)
        self._load_embeddings_and_lookup_dictionaries()

        self.all_labels = list(self.lbl2ent.keys())
        self.lbl2uri = self._load_label_to_uri()
        
        # Initialize EntityExtractor with embeddings lookup dictionaries
        # Property names not needed for recommender (pass empty list)
        self.entity_extractor = EntityExtractor(
            kg_handler=self.kg_handler,
            nlp=self.nlp,
            property_names=[],  # Not needed for recommender
            ent2lbl=self.ent2lbl if hasattr(self, 'ent2lbl') else None,
            all_entity_labels=self.all_entity_labels if hasattr(self, 'all_entity_labels') else None
        )
        
        # Initialize TF-IDF and CF components
        try:
            self._build_metadata_dataframe(use_cache=True, random_seed=42)
            self._build_similarity_matrix()
            self._build_collaborative_filtering()
        except Exception as e:
            print(f"[Error] initializing recommendation models: {e}")
            raise ValueError(f"Error initializing recommendation models.")
    
    def _load_label_to_uri(self):
        with open(LABEL_TO_URI_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def fetch_movie_metadata(self, item_uri: str) -> Dict[str, str]:
        """
        Fetches metadata for a movie from the knowledge graph.
        
        Args:
            item_uri: The Wikidata URI of the movie
            
        Returns:
            Dictionary with metadata fields (title, genres, directors, year, description, types)
        """
        metadata = {
            'item_id': item_uri,
            'title': '',
            'genres': '',
            'directors': '',
            'year': '',
            'description': '',
            'types': ''
        }
        
        # Get title/label
        try:
            title = self.kg_handler.get_label_for_uri(item_uri)
            metadata['title'] = title if title else ''
        except Exception as e:
            print(f"Warning: Could not get label for {item_uri}: {e}")
        
        # Get metadata using SPARQL query
        query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX schema: <http://schema.org/>
            
            SELECT DISTINCT ?genreLabel ?directorLabel ?year ?description ?typeLabel WHERE {{
                <{item_uri}> rdfs:label ?title .
                
                # Get genres (P136)
                OPTIONAL {{
                    <{item_uri}> wdt:P136 ?genre .
                    ?genre rdfs:label ?genreLabel .
                    FILTER(LANG(?genreLabel) = '' || LANGMATCHES(LANG(?genreLabel), 'en'))
                }}
                
                # Get directors (P57)
                OPTIONAL {{
                    <{item_uri}> wdt:P57 ?director .
                    ?director rdfs:label ?directorLabel .
                    FILTER(LANG(?directorLabel) = '' || LANGMATCHES(LANG(?directorLabel), 'en'))
                }}
                
                # Get publication date/year (P577)
                OPTIONAL {{
                    <{item_uri}> wdt:P577 ?date .
                    BIND(YEAR(?date) AS ?year)
                }}
                
                # Get description
                OPTIONAL {{
                    <{item_uri}> schema:description ?description .
                    FILTER(LANG(?description) = '' || LANGMATCHES(LANG(?description), 'en'))
                }}
                
                # Get instance types (P31)
                OPTIONAL {{
                    <{item_uri}> wdt:P31 ?type .
                    ?type rdfs:label ?typeLabel .
                    FILTER(LANG(?typeLabel) = '' || LANGMATCHES(LANG(?typeLabel), 'en'))
                }}
            }}
        """
        
        try:
            results = self.kg_handler.sparql_query(query)
            if isinstance(results, list) and results:
                genres = []
                directors = []
                years = []
                descriptions = []
                types = []
                
                for row in results:
                    if 'genreLabel' in row and row['genreLabel']:
                        genres.append(row['genreLabel'].lower())
                    if 'directorLabel' in row and row['directorLabel']:
                        directors.append(row['directorLabel'].lower().replace(' ', ''))
                    if 'year' in row and row['year']:
                        years.append(str(row['year']))
                    if 'description' in row and row['description']:
                        descriptions.append(row['description'].lower())
                    if 'typeLabel' in row and row['typeLabel']:
                        types.append(row['typeLabel'].lower())
                
                metadata['genres'] = ' '.join(list(set(genres)))
                metadata['directors'] = ' '.join(list(set(directors)))
                metadata['year'] = ' '.join(list(set(years)))
                metadata['description'] = ' '.join(list(set(descriptions)))
                metadata['types'] = ' '.join(list(set(types)))
        except Exception as e:
            print(f"[Metadata Fetch] Warning: SPARQL query failed for {item_uri}: {e}")
        
        # Fallback: try get_entity_metadata_local
        if not metadata['description'] and not metadata['types']:
            try:
                local_metadata = self.kg_handler.get_entity_metadata_local(item_uri)
                if local_metadata.get('description'):
                    metadata['description'] = local_metadata['description']
                if local_metadata.get('types'):
                    metadata['types'] = ' '.join(local_metadata['types'])
            except Exception as e:
                pass  # Silent fail, we already have what we can get
        
        return metadata
    
    def _get_cache_path(self, limit: Optional[int] = None, sample_size: Optional[int] = None, 
                        random_seed: Optional[int] = None) -> str:
        """
        Generate a cache file path based on the parameters and ratings file content.
        
        Args:
            limit: Maximum number of items to process
            sample_size: Sample size for random sampling
            random_seed: Random seed for reproducible sampling (required for caching with sample_size)
            
        Returns:
            Path to the cache file
        """
        # Create a hash based on parameters and ratings file paths
        # This ensures cache is invalidated if ratings files change
        cache_key = f"{self.user_ratings_path}_{self.item_ratings_path}_{limit}_{sample_size}_{random_seed}"
        
        # Include file modification times to detect changes
        try:
            user_ratings_mtime = os.path.getmtime(self.user_ratings_path)
            item_ratings_mtime = os.path.getmtime(self.item_ratings_path)
            cache_key += f"_{user_ratings_mtime}_{item_ratings_mtime}"
        except OSError:
            pass  # Files might not exist yet
        
        # Create hash of the cache key
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_filename = f"metadata_cache_{cache_hash}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_metadata_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Load metadata DataFrame from cache.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Cached DataFrame if successful, None otherwise
        """
        if not os.path.exists(cache_path):
            return None
        
        try:
            print(f"[Metadata Cache] Loading metadata from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify it's a DataFrame with expected columns
            if isinstance(cached_data, pd.DataFrame):
                expected_columns = ['item_id', 'title', 'genres', 'directors', 'year', 'description', 'types', 'soup']
                if all(col in cached_data.columns for col in expected_columns):
                    print(f"[Metadata Cache] Successfully loaded {len(cached_data)} items from cache")
                    return cached_data
                else:
                    print(f"[Metadata Cache] Cache file has unexpected structure, rebuilding...")
            else:
                print(f"[Metadata Cache] Cache file is not a DataFrame, rebuilding...")
        except Exception as e:
            print(f"[Metadata Cache] Error loading cache: {e}. Rebuilding...")
        
        return None
    
    def _save_metadata_cache(self, metadata_df: pd.DataFrame, cache_path: str):
        """
        Save metadata DataFrame to cache.
        
        Args:
            metadata_df: DataFrame to cache
            cache_path: Path to save the cache file
        """
        try:
            print(f"[Metadata Cache] Saving metadata to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(metadata_df, f)
            print(f"[Metadata Cache] Cache saved successfully")
        except Exception as e:
            print(f"[Metadata Cache] Warning: Could not save cache: {e}")
    
    def _build_metadata_dataframe(self, limit: Optional[int] = None, sample_size: Optional[int] = None, 
                                 use_cache: bool = True, force_rebuild: bool = False,
                                 random_seed: Optional[int] = None):
        """
        Builds a DataFrame with metadata for all movies in the ratings files.
        Uses caching to avoid rebuilding if the same parameters are used.
        
        Args:
            limit: Maximum number of items to process (None = all)
            sample_size: If specified, randomly sample this many items for faster prototyping
            use_cache: Whether to use cached metadata if available
            force_rebuild: If True, rebuild even if cache exists
            random_seed: Random seed for reproducible sampling. 
                        If sample_size is used without a seed, caching will not work reliably.
        """
        # Check cache first
        cache_path = self._get_cache_path(limit=limit, sample_size=sample_size, random_seed=random_seed)
        
        if use_cache and not force_rebuild:
            cached_df = self._load_metadata_cache(cache_path)
            if cached_df is not None:
                self.metadata_df = cached_df
                return self.metadata_df
        
        # Cache miss or force rebuild - fetch metadata
        print("[Metadata DataFrame] Loading ratings files...")
        user_ratings = pd.read_csv(self.user_ratings_path)
        item_ratings = pd.read_csv(self.item_ratings_path)
        
        # Get unique item IDs
        unique_items = user_ratings['item_id'].unique()
        if sample_size and sample_size < len(unique_items):
            print(f"[Metadata DataFrame] Sampling {sample_size} items from {len(unique_items)} total items...")
            if random_seed is not None:
                np.random.seed(random_seed)
                print(f"[Metadata DataFrame] Using random seed: {random_seed} for reproducible sampling")
            unique_items = np.random.choice(unique_items, size=sample_size, replace=False)
        elif limit:
            unique_items = unique_items[:limit]
        
        print(f"[Metadata DataFrame] Fetching metadata for {len(unique_items)} movies...")
        print("This may take a while...")
        
        metadata_list = []
        for item_uri in tqdm(unique_items, desc="Fetching metadata"):
            metadata = self.fetch_movie_metadata(item_uri)
            metadata_list.append(metadata)
        
        self.metadata_df = pd.DataFrame(metadata_list)
        
        # Create a 'soup' of features for each movie
        print("[Metadata DataFrame] Creating feature soup...")
        self.metadata_df['soup'] = (
            self.metadata_df['genres'] + ' ' +
            self.metadata_df['directors'] + ' ' +
            self.metadata_df['year'] + ' ' +
            self.metadata_df['types'] + ' ' +
            self.metadata_df['description']
        )
        self.metadata_df['soup'] = self.metadata_df['soup'].fillna('').str.strip()
        
        # Remove rows with empty soup (no metadata found)
        initial_count = len(self.metadata_df)
        self.metadata_df = self.metadata_df[self.metadata_df['soup'].str.len() > 0]
        removed_count = initial_count - len(self.metadata_df)
        if removed_count > 0:
            print(f"[Metadata DataFrame] Removed {removed_count} items with no metadata")
        
        # Check for duplicate titles (same label, different URIs)
        duplicate_titles = self.metadata_df[self.metadata_df.duplicated(subset=['title'], keep=False)]
        if not duplicate_titles.empty:
            duplicate_count = len(duplicate_titles['title'].unique())
            print(f"[Metadata DataFrame] Warning: Found {duplicate_count} titles that appear multiple times with different URIs")
        
        # Verify all URIs are unique (should always be true)
        duplicate_uris = self.metadata_df[self.metadata_df.duplicated(subset=['item_id'], keep=False)]
        if not duplicate_uris.empty:
            print(f"[Metadata DataFrame] ERROR: Found duplicate URIs in metadata! This should not happen.")
            print(f"[Metadata DataFrame] Duplicate URIs: {duplicate_uris['item_id'].unique()}")
        
        print(f"[Metadata DataFrame] Metadata DataFrame created with {len(self.metadata_df)} movies")
        
        # Save to cache
        if use_cache:
            # Warn if using sample_size without a seed (cache won't be reliable)
            if sample_size and random_seed is None:
                print(f"[Metadata DataFrame] Warning: Using sample_size without random_seed. "
                      f"Cache may not be reliable for future runs with different random samples.")
            self._save_metadata_cache(self.metadata_df, cache_path)
    
    def _build_similarity_matrix(self):
        """
        Builds the TF-IDF matrix and cosine similarity matrix.
        """
        if self.metadata_df is None:
            raise ValueError("Must build metadata DataFrame first. Call _build_metadata_dataframe()")
        
        print("[TF-IDF Matrix] Building TF-IDF matrix...")
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.metadata_df['soup'])
        
        print("[TF-IDF Matrix] Computing cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("[TF-IDF Matrix] Similarity matrix built successfully!")
    
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
                self.ent2lbl = {}
                self.lbl2ent = defaultdict(list)
                for ent, lbl in self.kg_handler.graph.subject_objects(RDFS.label):
                    ent_uri = rdflib.URIRef(ent)
                    lbl_str = str(lbl)
                    self.ent2lbl[ent_uri] = lbl_str
                    self.lbl2ent[lbl_str].append(ent_uri)
                print(f"[Embeddings] Loaded {len(self.ent2lbl)} entity2label entries and {len(self.lbl2ent)} label2entity entries")

                self.all_entity_labels = list(self.ent2lbl.values())
                print(f"[Embeddings] Loaded {len(self.all_entity_labels)} entity labels (all)")
            except Exception as e:
                print(f"[Embeddings] Error loading embeddings: {e}")
                raise e

        else:
            raise FileNotFoundError(f"[ERROR] Embeddings directory not found: {self.embeddings_path}. Please run the embedding script to generate the embeddings.")
    
    def resolve_entity_to_candidates(self, key):
        """
        Find all candidate entity URIs for a given label/key.
        Uses exact and fuzzy matching of labels to find all possible entity URIs.
        Returns a list of candidate URIs (can be empty).
        """
        # exact match first
        candidates = []
        if key in self.lbl2ent:
            candidates.extend(self.lbl2ent[key])  # extend with list contents
        # fuzzy match
        match = difflib.get_close_matches(key, self.all_labels, n=1, cutoff=0.7)
        for m in match:
            candidates.extend(self.lbl2ent[m])  # extend with each list's contents
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def resolve_entities(self, entities):
        """
        Resolve all entities to their candidate URIs.
        Returns a list of lists: [[uri1, uri2], [uri3], ...] where each inner list
        contains all candidate URIs for one entity.
        """
        liked_uris = []
        for e in entities:
            candidate_uris = self.resolve_entity_to_candidates(e["text"])
            liked_uris.append(candidate_uris)  # append list of candidates
        return liked_uris
    
    def recursively_collect_traits(self, uri, visited=None, depth=0):
        """
        Recursively collect (predicate, object) pairs of INTERESTING_PROPS,
        following RELATIONAL_PROPS up to MAX_DEPTH.
        Returns list of (source_uri, predicate, object).
        """
        if visited is None:
            visited = set()
        if uri in visited or depth > self.MAX_DEPTH:
            return []
        visited.add(uri)
        
        traits = []
        # Collect interesting traits for this URI
        for p, o in self.kg_handler.graph.predicate_objects(uri):
            if p in INTERESTING_PROPS:
                traits.append((uri, p, o))
        
        # Recurse through relational links to explore deeper
        for p, o in self.kg_handler.graph.predicate_objects(uri):
            if p in RELATIONAL_PROPS:
                traits.extend(self.recursively_collect_traits(o, visited, depth + 1))
        return traits
    
    def infer_common_traits_structured(self, candidates, min_count=1, depth=3):
        """
        Works with nested candidates: [[uri1, uri2, ...], [uri3, ...], ...]
        Returns list of dicts [{label: "", uri: "", property: "", count: "", sources: []}, {...}]:
        - label: rdfs:label of object
        - uri: object uri
        - property: predicate label(s)
        - count: number of distinct *source URIs* that led to this object
        - sources: list of URIs that contributed
        Filters by min_count, sorts by count desc then label.
        """
        self.MAX_DEPTH = depth

        # map: object -> set(source_uris)
        trait_sources = defaultdict(set)
        # map: object -> set(predicates encountered)
        trait_predicates = defaultdict(set)

        # Handle nested structure [[uri1, uri2], [uri3, uri4], ...]
        for entity_group in candidates:
            if not entity_group:
                continue  # skip empty entity lists
            for src_uri in entity_group:
                traits = self.recursively_collect_traits(src_uri)
                # Each trait is (source_uri, predicate, object)
                for _, p, o in traits:
                    trait_sources[o].add(src_uri)
                    trait_predicates[o].add(p)

        # Build structured result
        items = []
        for o, srcs in trait_sources.items():
            cnt = len(srcs)
            if cnt < min_count:
                continue
            # Get predicate labels
            preds = list(trait_predicates[o])
            pred_labels = []
            for p in preds:
                pl = self.kg_handler.graph.value(p, RDFS.label)
                pred_labels.append(str(pl) if pl else str(p))
            pred_label_joined = "; ".join(sorted(pred_labels))

            # Get object label (if any)
            label = self.kg_handler.graph.value(o, RDFS.label)

            items.append({
                "label": str(label) if label else None,
                "uri": str(o),
                "property": pred_label_joined,
                "count": cnt,
                "sources": [str(s) for s in sorted(srcs, key=lambda x: str(x))],
            })

        # Sort results: descending count, then label
        items.sort(key=lambda x: (-x["count"], x["label"] or ""))
        return items


    def is_film(self, uri):
        types = set(self.kg_handler.graph.objects(uri, WDT.P31))
        # check if it is a film or anything similar
        if types & FILM_CLASSES:
            return uri

    def filter_candidates(self, candidates):
        selected = []
        for entity in candidates:
            ent = []
            for uri in entity:
                if self.is_film(uri):
                    ent.append(uri)
            selected.append(ent)
        return selected
    
    def print_entities(self, entities, candidates, selected_candidates):
        print(f"\n{3*'-'}\nResolved Entities:")
        for i, cand in enumerate(candidates):
            sel_cand = selected_candidates[i]
            entity_name = entities[i]["text"]
            print(f"\n{entity_name}:")
            for uri in cand:
                get_type = next(iter(set(self.kg_handler.graph.objects(uri, WDT.P31))), None)
                type_label = self.ent2lbl.get(get_type, "Unknown")
                selected = "OK" if uri in sel_cand else "X"
                print("    {:<50} {:<30} {:<6}".format(str(uri), type_label, selected))
        print(f"{3*'-'}\n")
    
    def _build_collaborative_filtering(self):
        """
        Builds the collaborative filtering model using item-based approach.
        Creates a user-item matrix and computes item-item similarity based on user ratings.
        """
        print("[Collaborative Filtering] Building collaborative filtering model...")
        
        # Load ratings
        self.user_ratings_df = pd.read_csv(self.user_ratings_path)
        self.item_ratings_df = pd.read_csv(self.item_ratings_path)
        
        print(f"[Collaborative Filtering] Loaded {len(self.user_ratings_df)} user ratings")
        print(f"[Collaborative Filtering] Loaded {len(self.item_ratings_df)} item ratings")
        
        # Create user-item matrix
        # Rows = users, Columns = items, Values = ratings
        print("[Collaborative Filtering] Creating user-item matrix...")
        self.user_item_matrix = self.user_ratings_df.pivot_table(
            index='user_id', 
            columns='item_id',
            values='rating'
        )
        
        # Fill missing values with item ratings from item_ratings.csv
        # Create a mapping from item_id to rating
        item_rating_map = dict(zip(self.item_ratings_df['item_id'], self.item_ratings_df['rating']))
        mean_rating = self.item_ratings_df['rating'].mean()
        # Create a Series with item ratings for columns that exist in the matrix
        fill_values = pd.Series([item_rating_map.get(item_id, mean_rating) for item_id in self.user_item_matrix.columns], 
                                index=self.user_item_matrix.columns)
        
        print(f"[Collaborative Filtering] Filling {self.user_item_matrix.isna().sum().sum()} missing values")

        # Fill missing values with the corresponding item rating
        self.user_item_matrix = self.user_item_matrix.fillna(fill_values)
        
        print(f"[Collaborative Filtering] User-item matrix shape: {self.user_item_matrix.shape}")
        
        # Create mapping between item URIs and matrix indices
        self.item_uri_to_cf_index = {uri: idx for idx, uri in enumerate(self.user_item_matrix.columns)}
        self.cf_index_to_item_uri = {idx: uri for uri, idx in self.item_uri_to_cf_index.items()}
        
        # Compute item-item similarity based on user ratings
        # Items are similar if users rate them similarly
        print("[Collaborative Filtering] Computing item-item similarity matrix (this may take a while)...")
        item_ratings_matrix = self.user_item_matrix.T  # Transpose: rows = items, columns = users
        self.item_similarity_cf = cosine_similarity(item_ratings_matrix)
        print("[Collaborative Filtering] Collaborative filtering model built successfully!")
    
    def get_collaborative_filtering_recommendations(self, item_uris: List[str], top_n: int = 10) -> pd.DataFrame:
        """
        Get movie recommendations using collaborative filtering (item-based).
        Finds items that are similar based on user rating patterns.
        
        Note: URIs are used as unique identifiers. Multiple items may share the same
        title/label (e.g., remakes), but each has a unique URI. The returned DataFrame
        includes both title and item_id (URI) to distinguish items with duplicate titles.
        
        Args:
            item_uris: List of Wikidata URIs of movies the user likes
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies and their CF similarity scores.
            Columns: title, item_id, cf_similarity_score, genres, directors, year
        """
        if self.item_similarity_cf is None:
            raise ValueError("Must build collaborative filtering model first. Call build_collaborative_filtering()")
        
        # Find indices of input movies in the CF matrix
        # Convert URIs to strings for comparison (URIs may come as rdflib.term.URIRef objects)
        movie_cf_indices = []
        found_uris = []
        for uri in item_uris:
            uri_str = str(uri)  # Convert to string for comparison
            if uri_str in self.item_uri_to_cf_index:
                movie_cf_indices.append(self.item_uri_to_cf_index[uri_str])
                found_uris.append(uri_str)
            else:
                print(f"[Collaborative Filtering] Warning: Movie with URI '{uri_str}' not found in collaborative filtering matrix. Skipping...")
        
        if not movie_cf_indices:
            return pd.DataFrame(columns=['title', 'item_id', 'cf_similarity_score', 'genres', 'directors', 'year'])
        
        # Calculate average similarity scores
        if len(movie_cf_indices) == 1:
            sim_scores = list(enumerate(self.item_similarity_cf[movie_cf_indices[0]]))
        else:
            avg_sim_scores = self.item_similarity_cf[movie_cf_indices].mean(axis=0)
            sim_scores = list(enumerate(avg_sim_scores))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (excluding the input movies themselves)
        recommended_cf_indices = []
        recommended_scores = []
        
        for cf_idx, score in sim_scores:
            if cf_idx not in movie_cf_indices and score > 0:  # Exclude input movies and zero similarity
                recommended_cf_indices.append(cf_idx)
                recommended_scores.append(score)
                if len(recommended_cf_indices) >= top_n:
                    break
        
        # Convert CF indices back to URIs
        recommended_uris = [self.cf_index_to_item_uri[cf_idx] for cf_idx in recommended_cf_indices]
        
        # Get metadata for recommended items
        recommendations_list = []
        for uri, score in zip(recommended_uris, recommended_scores):
            # Try to get metadata from metadata_df if available
            if self.metadata_df is not None:
                matches = self.metadata_df[self.metadata_df['item_id'] == uri]
                if not matches.empty:
                    # Should only be one match since URIs are unique, but use iloc[0] to be safe
                    if len(matches) > 1:
                        print(f"[Collaborative Filtering] Warning: Found {len(matches)} rows with URI {uri}, using first match")
                    row = matches.iloc[0]
                    recommendations_list.append({
                        'item_id': uri,
                        'title': row['title'],
                        'cf_similarity_score': score,
                        'genres': row.get('genres', ''),
                        'directors': row.get('directors', ''),
                        'year': row.get('year', '')
                    })
                else:
                    # Item not in metadata_df, but we can still recommend it
                    # Try to get title from KG handler
                    title = ''
                    try:
                        title = self.kg_handler.get_label_for_uri(uri)
                    except:
                        pass
                    recommendations_list.append({
                        'item_id': uri,
                        'title': title if title else uri,
                        'cf_similarity_score': score,
                        'genres': '',
                        'directors': '',
                        'year': ''
                    })
            else:
                # No metadata_df, just get title from KG handler
                title = ''
                try:
                    title = self.kg_handler.get_label_for_uri(uri)
                except:
                    pass
                recommendations_list.append({
                    'item_id': uri,
                    'title': title if title else uri,
                    'cf_similarity_score': score,
                    'genres': '',
                    'directors': '',
                    'year': ''
                })
        
        recommendations = pd.DataFrame(recommendations_list)
        
        # Reorder columns to put item_id after title for clarity
        if not recommendations.empty:
            recommendations = recommendations[['title', 'item_id', 'cf_similarity_score', 'genres', 'directors', 'year']]
        
        return recommendations
    
    def _build_trait_query_vector(self, common_traits: List[Dict], top_n: int = 10) -> str:
        """
        Build a query string from common traits for TF-IDF querying.
        
        Args:
            common_traits: List of trait dictionaries from infer_common_traits_structured
            top_n: Number of top traits to include
            
        Returns:
            Query string with trait labels weighted by count
        """
        trait_query_parts = []
        for trait in common_traits[:top_n]:
            if trait.get('label'):
                # Repeat based on count to weight more common traits
                trait_query_parts.extend([trait['label']] * trait['count'])
        return ' '.join(trait_query_parts)
    
    def get_tfidf_recommendations_by_uri(self, item_uris: List[str], common_traits: List[Dict], 
                                         top_n: int = 10, trait_weight: float = 1.5) -> pd.DataFrame:
        """
        Get movie recommendations using TF-IDF with hybrid scoring based on common traits.
        
        Args:
            item_uris: List of Wikidata URIs of movies the user likes
            common_traits: Common traits from infer_common_traits_structured
            top_n: Number of recommendations to return
            trait_weight: Multiplier for movies matching common traits
            
        Returns:
            DataFrame with recommended movies and their similarity scores.
            Columns: title, item_id, similarity_score, genres, directors, year
        """
        if self.cosine_sim is None or self.tfidf_vectorizer is None:
            raise ValueError("Must build similarity matrix first. Call build_similarity_matrix()")
        
        # Find indices of input movies by URI
        # Convert URIs to strings for comparison (URIs may come as rdflib.term.URIRef objects)
        movie_indices = []
        for uri in item_uris:
            uri_str = str(uri)  # Convert to string for comparison
            matches = self.metadata_df[self.metadata_df['item_id'] == uri_str]
            if not matches.empty:
                movie_indices.append(matches.index[0])
            else:
                print(f"[TF-IDF Recommendations] Warning: Movie with URI '{uri_str}' not found in dataset. Skipping...")
        
        if not movie_indices:
            return pd.DataFrame(columns=['title', 'item_id', 'similarity_score', 'genres', 'directors', 'year'])
        
        # Calculate base similarity scores
        if len(movie_indices) == 1:
            sim_scores = list(enumerate(self.cosine_sim[movie_indices[0]]))
        else:
            avg_sim_scores = self.cosine_sim[movie_indices].mean(axis=0)
            sim_scores = list(enumerate(avg_sim_scores))
        
        # Build trait query and get trait-based similarity boost
        if common_traits:
            trait_query = self._build_trait_query_vector(common_traits, top_n=10)
            if trait_query:
                try:
                    trait_vector = self.tfidf_vectorizer.transform([trait_query])
                    trait_similarities = cosine_similarity(trait_vector, self.tfidf_matrix)[0]
                    
                    # Boost scores for movies matching common traits
                    for idx, (movie_idx, base_score) in enumerate(sim_scores):
                        trait_score = trait_similarities[movie_idx]
                        boosted_score = base_score * (1 + trait_weight * trait_score)
                        sim_scores[idx] = (movie_idx, boosted_score)
                except Exception as e:
                    print(f"[TF-IDF Recommendations] Warning: Could not apply trait-based boosting: {e}")
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommended_indices = []
        recommended_scores = []
        
        for idx, score in sim_scores:
            if idx not in movie_indices and score > 0:
                recommended_indices.append(idx)
                recommended_scores.append(score)
                if len(recommended_indices) >= top_n:
                    break
        
        # Create results DataFrame
        recommendations = self.metadata_df.iloc[recommended_indices][['title', 'item_id', 'genres', 'directors', 'year']].copy()
        recommendations['similarity_score'] = recommended_scores
        
        # Reorder columns to put item_id after title for clarity
        recommendations = recommendations[['title', 'item_id', 'similarity_score', 'genres', 'directors', 'year']]
        
        return recommendations
    
    def print_common_traits(self, common_traits, top_n=5, max_sources_per_line=5):
        """
        Print top N common traits in a tabular format.
        - Only show Q-IDs from URIs.
        - Wrap long source lists across multiple lines.
        """
        print(f"\n{3*'-'}\nTop Common Traits:")
        if not common_traits:
            print("  (none found)")
            return

        traits_to_show = common_traits[:top_n]

        # Header
        print("{:<25} {:<45} {:<8} {}".format("LABEL", "PROPERTY", "COUNT", "SOURCES"))
        print("-" * 100)

        for trait in traits_to_show:
            label = trait["label"] or "(no label)"
            prop = trait["property"]
            count = trait["count"]

            # Extract only Q-IDs
            sources = [re.sub(r".*/(Q\d+)$", r"\1", s) for s in trait["sources"]]

            # Wrap sources in chunks
            chunks = [", ".join(sources[i:i+max_sources_per_line]) for i in range(0, len(sources), max_sources_per_line)]

            # Print first line with label, property, count
            print("{:<25} {:<45} {:<8} {}".format(label[:25], prop[:45], count, chunks[0]))

            # Print remaining lines of sources indented
            for chunk in chunks[1:]:
                print("{:<25} {:<45} {:<8} {}".format("", "", "", chunk))
        print(f"{3*'-'}\n")

    def format_recommendations(self, tfidf_recommendations: pd.DataFrame, cf_recommendations: pd.DataFrame,
                              user_query: str, common_traits: List[Dict]) -> str:
        """
        Formats recommendations into natural language using LLM.
        
        Args:
            tfidf_recommendations: DataFrame with TF-IDF recommendations
            cf_recommendations: DataFrame with CF recommendations
            user_query: Original user query
            common_traits: Common traits for context
            
        Returns:
            Formatted natural language response
        """
        if self.llm_handler is None or self.prompt_manager is None:
            # Fallback to simple formatting if LLM handler not available
            return self._format_recommendations_simple(tfidf_recommendations, cf_recommendations, user_query)
        
        # Convert DataFrames to structured dict format
        tfidf_list = []
        for _, row in tfidf_recommendations.iterrows():
            tfidf_list.append({
                "title": row.get('title', ''),
                "item_id": row.get('item_id', ''),
                "similarity_score": float(row.get('similarity_score', 0)),
                "genres": row.get('genres', ''),
                "directors": row.get('directors', ''),
                "year": row.get('year', '')
            })
        
        cf_list = []
        for _, row in cf_recommendations.iterrows():
            cf_list.append({
                "title": row.get('title', ''),
                "item_id": row.get('item_id', ''),
                "cf_similarity_score": float(row.get('cf_similarity_score', 0)),
                "genres": row.get('genres', ''),
                "directors": row.get('directors', ''),
                "year": row.get('year', '')
            })
        
        try:
            # Get the prompt for recommendation formatting
            prompt = self.prompt_manager.get_prompt(
                "recommendation_formatter",
                user_query=user_query,
                tfidf_recommendations=json.dumps(tfidf_list, indent=2, ensure_ascii=False),
                cf_recommendations=json.dumps(cf_list, indent=2, ensure_ascii=False),
                common_traits=json.dumps(common_traits[:10] if common_traits else [], indent=2, ensure_ascii=False)
            )
            
            # Generate a natural language response using the LLM
            response = self.llm_handler.generate_response(prompt)
            
            if response['success']:
                # Check for speakeasy message limit
                if len(response['content']) > 1999:
                    return self._format_recommendations_simple(tfidf_recommendations, cf_recommendations, user_query)
                else:
                    return response['content']
            else:
                # Fallback to simple formatting if LLM fails
                return self._format_recommendations_simple(tfidf_recommendations, cf_recommendations, user_query)
                
        except Exception as e:
            print(f"[Recommendation Formatter] Error formatting recommendations with LLM: {e}")
            # Fallback to simple formatting if LLM fails
            return self._format_recommendations_simple(tfidf_recommendations, cf_recommendations, user_query)
    
    def _format_recommendations_simple(self, tfidf_recommendations: pd.DataFrame, 
                                       cf_recommendations: pd.DataFrame, user_query: str) -> str:
        """
        Simple fallback formatting without LLM.
        """
        result = "Based on your preferences, here are some recommendations:\n\n"
        
        if not tfidf_recommendations.empty:
            result += "Similar movies (content-based):\n"
            for _, row in tfidf_recommendations.head(5).iterrows():
                year = row.get('year', 'N/A')
                if year != 'N/A':
                    year = f" ({year})"
                result += f"- {row.get('title', 'Unknown')}{year}\n"
        
        if not cf_recommendations.empty:
            result += "\nOther users who like the same movies also like:\n"
            for _, row in cf_recommendations.head(5).iterrows():
                year = row.get('year', 'N/A')
                if year != 'N/A':
                    year = f" ({year})"
                result += f"- {row.get('title', 'Unknown')}{year}\n"

        result += "\nHave fun with the recommendations!"
        
        if tfidf_recommendations.empty and cf_recommendations.empty:
            result = "I couldn't find any recommendations based on your preferences. Please try with different movies."
        
        return result
    
    def _sanitize_message(self, message: str) -> str:
        """
        Sanitizes the message to remove common words and phrases that are not relevant to the recommendation.
        """
        message = message.replace("Recommend", "")
        message = message.replace("recommend", "")
        return message

    def recommend(self, message: str):
        """
        Returns a recommendation for a movie based on the user's query.
        """
        # 1. get the entities from the message/question based on different approaches
        # TODO: Check if fuzzy matching is beneficial for the recommender
        entities = self.entity_extractor.extract_entities(self._sanitize_message(message), use_fuzzy_match=True)
        print(f"[Movie Recommender] Entities: {entities}")

        if not entities:
            entities = self.entity_extractor.brute_force_extract_entities(self._sanitize_message(message))
            entities = entities[:3]
            print(f"[Movie Recommender] Brute force entities: {entities}")
        
        if not entities:
            return "I couldn't identify any movies in your message. Please mention specific movie titles you like."
        
        # 2(a). find the correct uris for the recommender from embeddings
        candidates = self.resolve_entities(entities)
        # 2(b). select the correct uris/entites from all possible candidates
        selected_candidates = self.filter_candidates(candidates)
        self.print_entities(entities, candidates, selected_candidates)
        
        # Flatten selected candidates to get list of URIs
        selected_uris = []
        for group in selected_candidates:
            selected_uris.extend(group)
        
        if not selected_uris:
            return "I couldn't find any valid movies in my database. Please try with different movie titles."
        
        # 3(a). get the "context" / common traits
        common_traits = self.infer_common_traits_structured(selected_candidates, min_count=1, depth=3)
        self.print_common_traits(common_traits, top_n=5, max_sources_per_line=3)
        
        # 4. Check if models are initialized
        if self.metadata_df is None or self.cosine_sim is None:
            return "I encountered an error while building the recommendation models. Please try again later."
        
        # 5. Get recommendations from both methods
        tfidf_recs = pd.DataFrame()
        cf_recs = pd.DataFrame()
        
        try:
            tfidf_recs = self.get_tfidf_recommendations_by_uri(selected_uris, common_traits, top_n=5)
            print(f"[Movie Recommender] TF-IDF recommendations: {tfidf_recs}")
        except Exception as e:
            print(f"[Movie Recommender] Warning: TF-IDF recommendations failed: {e}")
        
        try:
            cf_recs = self.get_collaborative_filtering_recommendations(selected_uris, top_n=5)
            print(f"[Movie Recommender] Collaborative filtering recommendations: {cf_recs}")
        except Exception as e:
            print(f"[Movie Recommender] Warning: Collaborative filtering recommendations failed: {e}")
        
        # 6. Format and return recommendations
        if tfidf_recs.empty and cf_recs.empty:
            return "I couldn't find any recommendations based on your preferences. Please try with different movies."
        else:
            return self.format_recommendations(tfidf_recs, cf_recs, message, common_traits)
