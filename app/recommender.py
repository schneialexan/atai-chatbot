import rdflib
import logging
import spacy
import difflib
import os
import csv
import json
import re

import pandas as pd
import numpy as np

from rdflib.namespace import RDFS
from collections import defaultdict

from app.kg_handler import LocalKnowledgeGraph

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
    def __init__(self, kg_handler: LocalKnowledgeGraph, dataset_path: str = "dataset", embeddings_path: str = "dataset/store/embeddings"):
        self.dataset_path = dataset_path
        self.kg_handler = kg_handler
        self.embeddings_path = embeddings_path
        self.nlp = spacy.load("en_core_web_trf")

        self.item_ratings = pd.read_csv(ITEM_RATINGS_PATH)
        self.user_ratings = pd.read_csv(USER_RATINGS_PATH)
        self._load_embeddings_and_lookup_dictionaries()

        self.all_labels = list(self.lbl2ent.keys())
        self.lbl2uri = self._load_label_to_uri()
    
    def _load_label_to_uri(self):
        with open(LABEL_TO_URI_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_embeddings_and_lookup_dictionaries(self):
        """Loads the embeddings and the lookup dictionaries."""
        if os.path.exists(self.embeddings_path):
            try:
                print("[Embeddings] Loading entity embeddings...")
                self.entity_emb = np.load(os.path.join(self.embeddings_path, "entity_embeds.npy"))
                
                print("[Embeddings] Loading relation embeddings...")
                self.relation_emb = np.load(os.path.join(self.embeddings_path, "relation_embeds.npy"))

                print("[Embeddings] Loading entity and relation IDs and preparing the embedding lookup dictionaries...")
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

    def get_entities(self, question):
        doc = self.nlp(question)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    def resolve_entity_to_film(self, key):
        """
        Find best matching film entity for a key.
        Uses fuzzy matching of labels + instance-of film classes.
        """
        # exact match first
        candidates = []
        if key in self.lbl2ent:
            candidates.append(self.lbl2ent[key])
        # fuzzy match
        match = difflib.get_close_matches(key, self.all_labels, n=1, cutoff=0.7)
        candidates.extend(self.lbl2ent[m] for m in match)

        for c in candidates:
            types = set(self.kg_handler.graph.objects(c, WDT.P31))
            # check if it is a film or anything similar
            if types & FILM_CLASSES:
                return c
            else:
                try:
                    return self.lbl2uri.get(self.ent2lbl[c])
                except:
                    continue
        # final fallback: return first candidate anyway
        return candidates[0] if candidates else None
    
    def resolve_entities(self, entities):
        liked_uris = []
        for e in entities:
            film_uri = self.resolve_entity_to_film(e["text"])
            liked_uris.append(film_uri)
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
        print("Resolved Entities:")
        for i, cand in enumerate(candidates):
            sel_cand = selected_candidates[i]
            entity_name = entities[i]["text"]
            print(f"\n{entity_name}:")
            for uri in cand:
                get_type = next(iter(set(self.kg_handler.graph.objects(uri, WDT.P31))), None)
                type_label = self.ent2lbl.get(get_type, "Unknown")
                selected = "OK" if uri in sel_cand else "X"
                print("    {:<50} {:<30} {:<6}".format(str(uri), type_label, selected))
    
    def print_common_traits(self, common_traits, top_n=5, max_sources_per_line=5):
        """
        Print top N common traits in a tabular format.
        - Only show Q-IDs from URIs.
        - Wrap long source lists across multiple lines.
        """
        print("\nTop Common Traits:")
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

    def recommend(self, message: str):
        """
        Returns a recommendation for a movie based on the user's query.
        """
        # 1. get the entities from the message/question via ner
        entities = self.get_entities(message)
        print(f"Entities: {entities}")
        # 2(a). find the correct uris for the recommender from embeddings
        candidates = self.resolve_entities(entities)
        # 2(b). select the correct uris/entites from all possible candidates
        selected_candidates = self.filter_candidates(candidates)
        self.print_entities(entities, candidates, selected_candidates)
        # 3(a). get the "context" / common traits
        common_traits = self.infer_common_traits_structured(selected_candidates, min_count=1, depth=3)
        self.print_common_traits(common_traits, top_n=5, max_sources_per_line=3)
        # 3(b). get the recommendation (even tough it is shit)
        # TODO
        return "I currently cannot recommend movies to you! I will be able to do this in the future!"
