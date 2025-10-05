# core.py
from .sparql_handler import LocalSPARQL
from .qa_handler import FactualQA
from .recommender import MovieRecommender
from .multimedia_handler import MultimediaHandler

class App:
    def __init__(self, dataset_path: str = "dataset", mode: int = "5"):
        self.dataset_path = dataset_path
        self.mode = mode
        self.sparql_handler = LocalSPARQL(dataset_path=dataset_path)
        self.qa_model = FactualQA(dataset_path=dataset_path)
        self.recommender = MovieRecommender(dataset_path=dataset_path)
        self.multimedia = MultimediaHandler(dataset_path=dataset_path)

        if mode not in [1, 2, 3, 4, 5]:
            raise ValueError("Mode must be one of: 1 (SPARQL), 2 (QA/embedding), 3 (recommendation), 4 (multimedia), 5 (auto)")

    def post_message(self, message: str):
        if self.mode == 1:
            return self.sparql_handler.query(message)
        elif self.mode == 2:
            return self.qa_model.answer(message)
        elif self.mode == 3:
            return self.recommender.recommend(message)
        elif self.mode == 4:
            return self.multimedia.get_image(message)
        elif self.mode == 5:
            return self._handle_all(message)

    def _handle_all(self, message: str):
        low = message.strip().lower()
        if low.startswith("select") or low.startswith("prefix") or "where {" in low:
            return self.sparql_handler.query(message)
        if "recommend" in low or "i like" in low:
            return self.recommender.recommend(message)
        if "picture" in low or "image" in low or "look like" in low:
            return self.multimedia.get_image(message)
        return self.qa_model.answer(message)
