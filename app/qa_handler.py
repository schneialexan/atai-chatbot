# qa_handler.py
import re
import logging
from typing import List, Optional, Any

from .sparql_handler import query_sparql, HEADERS

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("app.log", mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.handlers.clear()
logger.addHandler(file_handler)


# Optional: sentence-transformers for embedding fallback
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

# Optional: transformers for local LLM to translate NL -> SPARQL
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _HAS_T5 = True
except Exception:
    _HAS_T5 = False

# Prompt template used to instruct a local LLM to produce a SPARQL query for an input question.
LLM_PROMPT_TEMPLATE = """
You are an assistant that translates natural language questions into SPARQL queries for a target SPARQL endpoint.
Do NOT include any commentary — only output the SPARQL query.
Use these prefixes at the top of every query:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

Rules:
- Prefer label-based matching in English for entities (rdfs:label "..."@en).
- If the question expects a single compact answer (who, when, director) use LIMIT 1.
- If the question expects multiple answers (list, which), use LIMIT 10.
- Include: SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
- Do not invent Q-ids or property ids; use label matches (the SPARQL engine will resolve).
- If the question asks for a property you don't know, attempt a reasonable property mapping (e.g., director -> wdt:P57, genre -> wdt:P136, screenwriter -> wdt:P58, rating -> wdt:P1657) but you may rely on label matching instead.
- Keep queries dataset-agnostic as much as possible (they should work on Wikidata but also on other SPARQL endpoints with rdfs:label).

Examples:
Q: Who is the director of Star Wars: Episode VI - Return of the Jedi?
SPARQL:
SELECT ?answerLabel WHERE {
  ?entity rdfs:label "Star Wars: Episode VI - Return of the Jedi"@en .
  ?entity wdt:P57 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 1

Q: What is the genre of Good Neighbors?
SPARQL:
SELECT ?answerLabel WHERE {
  ?entity rdfs:label "Good Neighbors"@en .
  ?entity wdt:P136 ?answer .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 3

Now produce a SPARQL query for the question below — output only the SPARQL query (no explanation).
Question:
"""

# Default small local model name (changeable)
DEFAULT_T5 = "google/flan-t5-small"


class FactualQA:
    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql",
                 t5_model_name: str = DEFAULT_T5,
                 embed_model_name: str = "all-MiniLM-L6-v2"):
        self.endpoint_url = endpoint_url

        # a lightweight property map used as fallback only (not required if LLM present)
        self.property_map = {
            "director": "wdt:P57",
            "screenwriter": "wdt:P58",
            "writer": "wdt:P58",
            "genre": "wdt:P136",
            "rating": "wdt:P1657",
            "cast": "wdt:P161",
            "actor": "wdt:P161",
            "composer": "wdt:P86",
            "cinematographer": "wdt:P344",
            "producer": "wdt:P162"
        }

        # Load local LLM (seq2seq) if available
        self.llm_tokenizer = None
        self.llm_model = None
        if _HAS_T5:
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
            except Exception as e:
                logger.warning("Could not load local LLM model (%s): %s", t5_model_name, e)
                self.llm_tokenizer = None
                self.llm_model = None

        # Load embedding model if available
        self.embed_model = None
        if _HAS_SBERT:
            try:
                self.embed_model = SentenceTransformer(embed_model_name)
            except Exception as e:
                logger.warning("Could not load sentence-transformers model '%s': %s", embed_model_name, e)
                self.embed_model = None

    def _extract_title(self, question: str) -> Optional[str]:
        # Prefer quoted title "The Lion King"
        quoted = re.findall(r'"([^"]+)"', question)
        if quoted:
            return quoted[0].strip()

        # common pattern "Who is the director of X?" -> extract after 'of'
        m = re.search(r'of\s+([A-Z][^\?]+)', question)
        if m:
            candidate = m.group(1).strip().rstrip('?.')
            return candidate

        # fallback: last title-cased run
        caps = re.findall(r"[A-Z][a-z0-9'’\-]+(?:\s[A-Z][a-z0-9'’\-]+)*", question)
        if caps:
            return caps[-1].strip()
        return None

    def _call_llm_for_sparql(self, question: str) -> Optional[str]:
        if not (self.llm_tokenizer and self.llm_model):
            return None
        prompt = LLM_PROMPT_TEMPLATE + question + "\nSPARQL:\n"
        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True)
            out = self.llm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
            decoded = self.llm_tokenizer.decode(out[0], skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:
            logger.exception("LLM SPARQL generation failed: %s", e)
            return None

    def _query_sparql_raw(self, query: str):
        """
        Execute a SPARQL query and return the raw JSON. This is a thin wrapper around query_sparql.
        """
        # reuse query_sparql with a small timeout and retries
        data = query_sparql(self.endpoint_url, query, timeout=30, retries=3)
        # query_sparql returns parsed results; here we want the parsed output
        # but for convenience we will re-run a raw requests call to get the original JSON
        # (we keep it simple and reuse the parsed results instead)
        return {"results": {"bindings": [{"_placeholder": v} for v in data]}} if isinstance(data, list) else {"results": {"bindings": []}}

    def _run_property_query(self, title: str, prop: str, limit: int = 10) -> List[str]:
        """
        Attempt to fetch property values for an entity identified by label 'title'.
        Uses the fast EntitySearch when possible to avoid heavy label scans.
        """
        # Prefer using the MW API entity search service (fast on Wikidata)
        query = f"""
        SELECT ?answerLabel WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch" .
            bd:serviceParam wikibase:endpoint "www.wikidata.org" .
            bd:serviceParam mwapi:search "{title}" .
            bd:serviceParam mwapi:language "en" .
            ?item wikibase:apiOutputItem mwapi:item .
          }}
          ?item {prop} ?answer .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """
        try:
            # Use the HTTP call directly (lightweight)
            # We'll use query_sparql to get parsed results
            parsed = query_sparql(self.endpoint_url, query, timeout=30, retries=3)
            # query_sparql returns list of strings or tuples; convert to list of strings
            if not parsed:
                return []
            # If rows are tuples, take first element
            answers = []
            for row in parsed:
                if isinstance(row, tuple):
                    answers.append(row[0])
                else:
                    answers.append(row)
            return answers
        except Exception as e:
            logger.exception("SPARQL property query failed: %s", e)
            return []

    def _fetch_candidate_answers_for_entity(self, title: str, props: Optional[List[str]] = None, limit=50):
        """
        Gather candidate answers for an entity across multiple properties, used for embedding fallback.
        """
        if props is None:
            props = list(self.property_map.values())

        answers = []
        for p in props:
            vals = self._run_property_query(title, p, limit=min(limit, 10))
            for v in vals:
                if v not in answers:
                    answers.append(v)
        return answers

    def _semantic_answer(self, question: str, title: Optional[str]) -> Any:
        if not self.embed_model:
            return "No embedding model available to perform semantic fallback."

        if title:
            candidates = self._fetch_candidate_answers_for_entity(title, limit=50)
            if not candidates:
                return f"Sorry, I couldn't find candidate answers in the KG for '{title}'."

            q_emb = self.embed_model.encode(question, convert_to_tensor=True)
            cand_emb = self.embed_model.encode(candidates, convert_to_tensor=True)
            hits = sbert_util.semantic_search(q_emb, cand_emb, top_k=3)
            best = hits[0][0] if hits and hits[0] else None
            if best:
                return candidates[best["corpus_id"]]
            return "I couldn't find a semantically matching answer."
        else:
            # global candidate fetch is expensive; limit to a small sample if used
            sparql = """
            SELECT ?filmLabel ?genreLabel ?directorLabel WHERE {
              ?film wdt:P31 wd:Q11424 .
              OPTIONAL { ?film wdt:P136 ?genre. }
              OPTIONAL { ?film wdt:P57 ?director. }
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }
            LIMIT 200
            """
            try:
                parsed = query_sparql(self.endpoint_url, sparql, timeout=30, retries=2)
                candidates = []
                for row in parsed:
                    if isinstance(row, tuple):
                        candidates.extend([c for c in row if c])
                    else:
                        candidates.append(row)
                candidates = list(dict.fromkeys(candidates))
                if not candidates:
                    return "No candidates available for semantic search."
                q_emb = self.embed_model.encode(question, convert_to_tensor=True)
                cand_emb = self.embed_model.encode(candidates, convert_to_tensor=True)
                hits = sbert_util.semantic_search(q_emb, cand_emb, top_k=1)
                best = hits[0][0] if hits and hits[0] else None
                if best:
                    return candidates[best["corpus_id"]]
                return "No good embedding-based answer found."
            except Exception as e:
                logger.exception("Global embedding search failed: %s", e)
                return "Embedding search failed due to an error."

    def answer(self, question: str, use_embedding_fallback: bool = True):
        """
        Decide whether to run SPARQL (factual) or LLM->SPARQL or embedding fallback.
        Behavior:
          - If local LLM present: generate SPARQL from question and run it.
          - Else: if the question includes a known property keyword, run that property query.
          - Else: use embedding fallback (if available).
        """
        if not question or not question.strip():
            return "Empty question."

        question = question.strip()

        # 1) If LLM is available, prefer using it to produce a SPARQL query
        if self.llm_model and self.llm_tokenizer:
            sparql = self._call_llm_for_sparql(question)
            if sparql:
                try:
                    parsed = query_sparql(self.endpoint_url, sparql, timeout=30, retries=3)
                    if parsed:
                        # if a single scalar returned, return it; else return list
                        return parsed[0] if len(parsed) == 1 else parsed
                    else:
                        # If SPARQL returned nothing, try embedding fallback
                        if use_embedding_fallback and self.embed_model:
                            title = self._extract_title(question)
                            return self._semantic_answer(question, title)
                        return "No results from SPARQL."
                except Exception as e:
                    logger.exception("LLM-generated SPARQL failed: %s", e)
                    # fall back to embedding or property approach below

        # 2) No LLM result / LLM unavailable: check for property keywords
        lq = question.lower()
        for keyword, wikidata_prop in self.property_map.items():
            if keyword in lq:
                title = self._extract_title(question)
                if not title:
                    return "Sorry, I couldn't extract an entity title from your question."

                answers = self._run_property_query(title, wikidata_prop, limit=10)
                if answers:
                    return answers[0] if len(answers) == 1 else answers
                # If nothing found in SPARQL but embeddings available, try that
                if use_embedding_fallback and self.embed_model:
                    return self._semantic_answer(question, title)
                return f"Sorry, I could not find {keyword} information for {title}."

        # 3) Otherwise: treat as embedding-style question
        if use_embedding_fallback and self.embed_model:
            title = self._extract_title(question)
            return self._semantic_answer(question, title)

        return "Sorry, I couldn't process your question (no LLM and no embeddings available)."
