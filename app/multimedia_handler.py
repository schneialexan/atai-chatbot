# app/multimedia_handler.py
import os

class MultimediaHandler:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        # preload mapping actor/movie -> image file or URL
        self.image_index = self._load_index()

    def _load_index(self):
        # Here you’d map entity names/IDs to available images or imdb links
        # Could load from a CSV or JSON provided in dataset/additional
        return {}

    def get_image(self, query: str) -> str:
        """
        Returns an image link or path based on query (e.g. 'Show me a picture of Halle Berry').
        """
        for name, url in self.image_index.items():
            if name.lower() in query.lower():
                return url
        return "Sorry, I could not find an image for that entity."
