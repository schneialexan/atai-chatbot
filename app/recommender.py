# recommender.py
import os
# try to silence HF / tqdm progressbars early
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import logging
from typing import List, Dict, Set, Any
from collections import defaultdict

from .sparql_handler import query_sparql

logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    Faster recommender: batch fetches genres for all liked titles, then batch fetches movies for those genres.
    Robust parsing of different query_sparql return shapes (tuple/list/dict/str).
    """

    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        # we intentionally do NOT auto-load sentence-transformers here to avoid long startup and progressbars.
        # If you want embedding-based ranking later, add an explicit loader and pass show_progress_bar=False to encode.
        self.embed_model = None

    # --- utilities -----------------------------------------------------
    def _escape_literal(self, s: str) -> str:
        # minimal SPARQL literal escaping (quotes and backslashes)
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _query(self, query: str, timeout: int = 15, retries: int = 2) -> List[Any]:
        try:
            return query_sparql(self.endpoint_url, query, timeout=timeout, retries=retries)
        except Exception as e:
            # log warning (goes to file if you configured logging that way)
            logger.warning("SPARQL request failed: %s", e)
            return []

    def _parse_title_genre_rows(self, rows: List[Any]) -> Dict[str, Set[str]]:
        """
        Parse rows for SELECT ?title ?genreLabel ... returning a mapping title -> set(genres)
        Works with rows returned as tuples, dicts (SPARQL JSON parsed), or strings.
        """
        mapping = defaultdict(set)
        if not rows:
            return mapping

        for r in rows:
            title = None
            genre = None
            # tuple/list style (e.g., [('uri','Label'), ...] or ('Title','Genre'))
            if isinstance(r, (tuple, list)):
                strings = [x for x in r if isinstance(x, str) and not x.startswith("http")]
                if len(strings) >= 2:
                    title, genre = strings[0], strings[1]
            # dict style (e.g., {'titleLiteral': {'value': '...'}, 'genreLabel': {'value': '...'}})
            elif isinstance(r, dict):
                for k, v in r.items():
                    val = v.get("value") if isinstance(v, dict) and "value" in v else v
                    if not isinstance(val, str):
                        continue
                    k_l = k.lower()
                    if "title" in k_l:
                        title = val
                    elif "genre" in k_l:
                        genre = val
                # sometimes keys might be filmLabel / genreLabel; above covers that
            # single string rows (rare for pairs) - skip
            if title and genre:
                mapping[title].add(genre)
        return mapping

    def _parse_label_rows(self, rows: List[Any]) -> List[str]:
        """
        Parse rows that return a single label variable (recLabel / filmLabel / answerLabel).
        Return deduplicated list preserving order.
        """
        out = []
        if not rows:
            return out

        for r in rows:
            label = None
            if isinstance(r, (tuple, list)):
                # pick first human-readable string (skip URIs)
                strings = [x for x in r if isinstance(x, str) and not x.startswith("http")]
                if strings:
                    label = strings[0]
            elif isinstance(r, dict):
                # find any key that looks like a label and grab its value
                for k, v in r.items():
                    if isinstance(v, dict) and "value" in v:
                        if any(kword in k.lower() for kword in ("label", "answer", "film", "rec", "title")):
                            label = v["value"]
                            break
                        # fallback: take first string value found
                        if label is None:
                            label = v["value"]
                # If still None, try turning dict items into strings
            elif isinstance(r, str):
                label = r

            if label and label not in out:
                out.append(label)
        return out

    # --- SPARQL batches ------------------------------------------------
    def _batch_genres_for_titles(self, titles: List[str]) -> Dict[str, Set[str]]:
        """
        Get genres for multiple titles in a single query.
        Returns mapping: title -> set(genres)
        """
        if not titles:
            return {}

        values = " ".join(f'"{self._escape_literal(t)}"@en' for t in titles)
        q = f"""
        SELECT DISTINCT ?titleLiteral ?genreLabel WHERE {{
          VALUES ?titleLiteral {{ {values} }}
          ?movie rdfs:label ?titleLiteral .
          ?movie wdt:P136 ?genre .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 200
        """
        rows = self._query(q)
        mapping = self._parse_title_genre_rows(rows)

        # If mapping empty, we'll try a fuzzy-ish batch fallback below (in recommend)
        return mapping

    def _batch_movies_for_genres(self, genres: List[str], limit: int = 60) -> List[str]:
        """
        Fetch movies for many genres in one shot.
        """
        if not genres:
            return []

        values = " ".join(f'"{self._escape_literal(g)}"@en' for g in genres)
        q = f"""
        SELECT DISTINCT ?recLabel ?glabel WHERE {{
          VALUES ?glabel {{ {values} }}
          ?genre rdfs:label ?glabel .
          ?rec wdt:P136 ?genre .
          ?rec wdt:P31 wd:Q11424 .   # instance of film
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """
        rows = self._query(q)
        recs = self._parse_label_rows(rows)
        return recs

    # --- high-level recommend -----------------------------------------
    def _build_contains_filter(self, titles: List[str]) -> str:
        # Build a single FILTER(CONTAINS(... ) || CONTAINS(...) || ...)
        conds = []
        for t in titles:
            t_esc = self._escape_literal(t.lower())
            # use LCASE on filmLabel
            conds.append(f'CONTAINS(LCASE(?filmLabel), "{t_esc}")')
        return " || ".join(conds)

    def recommend(self, message: str) -> str:
        # naive extraction (same approach you had originally)
        candidates = re.findall(r"[A-Z][a-z0-9'’\-]+(?:\s[A-Z][a-z0-9'’\-]+)*", message)
        titles = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                titles.append(c)

        if not titles:
            return "I could not extract any movie titles from your request."

        # 1) Batch fetch genres for all provided titles
        mapping = self._batch_genres_for_titles(titles)

        # 1a) If no exact label matches found, try a single fuzzy-batch search (cheaper than per-title fuzzy)
        if not mapping:
            logger.debug("No exact-label genres found, running fuzzy batch search for titles: %s", titles)
            filter_cond = self._build_contains_filter(titles)
            if filter_cond:
                q_fuzzy = f"""
                SELECT DISTINCT ?filmLabel ?genreLabel WHERE {{
                  ?film wdt:P31 wd:Q11424 .
                  ?film rdfs:label ?filmLabel .
                  OPTIONAL {{ ?film wdt:P136 ?genre . ?genre rdfs:label ?genreLabel@en. }}
                  FILTER ( {filter_cond} )
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT 200
                """
                rows = self._query(q_fuzzy)
                # parse similarly - map filmLabel -> genreLabel
                parsed_map = defaultdict(set)
                if rows:
                    for r in rows:
                        title = None
                        genre = None
                        if isinstance(r, (tuple, list)):
                            strings = [x for x in r if isinstance(x, str) and not x.startswith("http")]
                            if strings:
                                title = strings[0]
                                genre = strings[1] if len(strings) > 1 else None
                        elif isinstance(r, dict):
                            for k, v in r.items():
                                val = v.get("value") if isinstance(v, dict) and "value" in v else v
                                if not isinstance(val, str):
                                    continue
                                if "filmlabel" in k.lower() or "title" in k.lower():
                                    title = val
                                elif "genre" in k.lower():
                                    genre = val
                        if title and genre:
                            parsed_map[title].add(genre)
                mapping = parsed_map

        # collect union of genres
        genres_set = set()
        for gset in mapping.values():
            genres_set.update(gset)

        if not genres_set:
            # final fallback: a very small generic genre-to-movies query to return *something*
            logger.debug("No genres found for provided titles, attempting a generic popular-movies fallback.")
            fallback_q = """
            SELECT DISTINCT ?recLabel WHERE {
              ?rec wdt:P31 wd:Q11424 .
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 30
            """
            rows = self._query(fallback_q)
            recs = self._parse_label_rows(rows)
            # remove liked titles
            recs = [r for r in recs if r not in titles]
            if not recs:
                return "Sorry, I could not find recommendations."
            return "Recommended movies: " + ", ".join(recs[:10])

        # 2) Batch fetch movies for all genres we collected
        recs = self._batch_movies_for_genres(list(genres_set), limit=120)

        # 3) clean, dedupe, remove liked titles
        recommended = [r for r in recs if r not in titles]
        # preserve order, dedupe
        recommended = list(dict.fromkeys(recommended))

        if not recommended:
            return "Sorry, I could not find recommendations."

        top = recommended[:10]
        return "Recommended movies: " + ", ".join(top)
