# core.py
from .sparql_handler import query_sparql
from .qa_handler import FactualQA
from .recommender import MovieRecommender

class App:
    """
    App orchestrator. mode:
      1 - raw SPARQL (message must be SPARQL)
      2 - factual / embedding / LLM->SPARQL
      3 - recommendation
      4 - autodetect (SPARQL if message looks like SPARQL, "recommend" if contains recommend, else QA)
    """
    def __init__(self, endpoint_url: str, mode: int):
        self.endpoint_url = endpoint_url
        self.mode = mode
        self.qa_model = FactualQA(endpoint_url=endpoint_url)
        self.recommender = MovieRecommender(endpoint_url=endpoint_url)

        if mode not in [1, 2, 3, 4]:
            raise ValueError("Mode must be one of: 1 (SPARQL), 2 (factual/embedding), 3 (recommendation), 4 (all)")

    def post_message(self, message: str):
        if self.mode == 1:
            # raw SPARQL
            return query_sparql(self.endpoint_url, message)
        elif self.mode == 2:
            return self.qa_model.answer(message)
        elif self.mode == 3:
            return self.recommender.recommend(message)
        elif self.mode == 4:
            return self._handle_all(message)

    def _handle_all(self, message: str):
        low = message.strip().lower()
        # heuristic: treat as SPARQL if it begins with SELECT/PREFIX/ASK or contains "where {"
        if low.startswith("select") or low.startswith("prefix") or "where {" in low:
            return query_sparql(self.endpoint_url, message)
        if "recommend" in low or "i like" in low:
            return self.recommender.recommend(message)
        return self.qa_model.answer(message)
