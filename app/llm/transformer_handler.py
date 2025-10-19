import os
from huggingface_hub import hf_hub_download

# Conditional imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModel
    from sentence_transformers import SentenceTransformer
except ImportError:
    pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModel, SentenceTransformer = None, None, None, None, None

class TransformerHandler:
    """A handler for a single transformer-based model."""

    def __init__(self, model_repo: str, model_dir: str = "models",
                 auto_load: bool = True, model_type: str = None, **kwargs):
        print(f"Initializing TransformerHandler for model: {model_repo}")
        self.model_repo = model_repo
        self.model_dir = model_dir
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.embedding_model = None

        if model_type is None:
            raise ValueError("'model_type' is required for transformer models.")
        if model_type is not None and AutoTokenizer is None:
            raise ImportError("transformers is not installed.")
        
        # Load model during initialization if auto_load is True
        if auto_load:
            self._load_model()
    
    def __del__(self):
        """Destructor to properly unload the model when the object is destroyed."""
        self.unload_model()

    def _load_model(self):
        print(f"Loading '{self.model_type}' model: {self.model_repo}...")
        
        if self.model_type == "embedding":
            # Load sentence transformer model for embeddings
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed.")
            self.embedding_model = SentenceTransformer(self.model_repo)
            print(f"Embedding model: {self.model_repo} loaded successfully.")
        elif self.model_type == "ner":
            # Load NER model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_repo)
            self.model = pipeline(self.model_type, model=self.model, tokenizer=self.tokenizer)
            print(f"NER model: {self.model_repo} loaded successfully.")
        else:
            # Default behavior for other model types
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_repo)
            self.model = pipeline(self.model_type, model=self.model, tokenizer=self.tokenizer)
            print(f"Model: {self.model_repo} loaded successfully.")
    
    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is not None or self.embedding_model is not None:
            print(f"Unloading model: {self.model_repo}")
            self.model = None
            self.tokenizer = None
            self.embedding_model = None
            print("Model unloaded successfully.")
    
    def load_model(self):
        """Manually load the model if it wasn't loaded during initialization."""
        if self.model is None and self.embedding_model is None:
            self._load_model()
    
    def is_loaded(self):
        """Check if the model is currently loaded."""
        return self.model is not None or self.embedding_model is not None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically unload model."""
        self.unload_model()

    def generate_ner_response(self, text: str, aggregation_strategy: str = "simple") -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first or set auto_load=True during initialization.")
        
        ner_results = self.model(text, aggregation_strategy=aggregation_strategy)
        return {
            'raw_response': ner_results,
            'content': ner_results,
            'success': True
        }
    
    def generate_embedding(self, texts: list) -> list:
        """Generate embeddings for a list of texts using the sentence transformer model."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not loaded. Call load_model() first or set auto_load=True during initialization.")
        
        if not isinstance(texts, list):
            texts = [texts]
        
        try:
            # Generate embeddings using sentence transformer
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []
