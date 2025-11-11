# core.py
from .kg_handler import LocalKnowledgeGraph
from .qa_handler import QAHandler
from .recommender import MovieRecommender
from .multimedia_handler import MultimediaHandler
from .llm.llm_handler import LLMHandler
from .llm.prompt_manager import PromptManager
from config import LLM_CONFIG

class App:
    def __init__(self, dataset_path: str = "dataset", embeddings_path: str = "dataset/store/embeddings", preload_strategy: str = "none", mode: int = None):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.preload_strategy = preload_strategy
        self.mode = mode
        self.prompt_manager = PromptManager()
        # Lazy initialization - models will be created only when needed
        self._kg_handler = None
        self._qa_model = None
        self._recommender = None
        self._multimedia = None
        self._llm_handlers = {}

        # Preload intent classifier
        self.intent_classifier = self._get_llm_handler("intent_classifier")
        
        # Apply preload strategy
        if self.preload_strategy == "all":
            print("Preloading all models...")
            self._preload_all_models()
        elif self.preload_strategy == "mode_specific" and self.mode is not None:
            print(f"Preloading models for mode {self.mode}...")
            self.preload_models_for_mode(self.mode)
        elif self.preload_strategy == "none":
            print("No preloading of models...")
        else:
            raise ValueError("Invalid preload strategy")

    def _preload_all_models(self):
        """Preload all models for production use to avoid first-message delay"""
        print("Preloading models for production use...")
        # Initialize all models in the background
        self._get_kg_handler()
        self._get_qa_model()
        self._get_recommender()
        self._get_multimedia()
        print("All models preloaded successfully!")

    def preload_models_for_mode(self, mode: int):
        """Preload only the models needed for a specific mode"""
        if mode == 1:
            self._get_kg_handler()
        elif mode == 2:
            self._get_qa_model()
        elif mode == 3:
            self._get_recommender()
        elif mode == 4:
            self._get_multimedia()
        elif mode == 5:
            # For auto mode, preload all models
            self._preload_all_models()

    def _get_kg_handler(self):
        """Lazy initialization of Knowledge Graph handler"""
        if self._kg_handler is None:
            self._kg_handler = LocalKnowledgeGraph(dataset_path=self.dataset_path)
        return self._kg_handler

    def _get_llm_handler(self, handler_name: str):
        """Lazy initialization of a named LLM handler."""
        if handler_name not in self._llm_handlers:
            print(f"Initializing LLM Handler for {handler_name}...")
            config = LLM_CONFIG.get(handler_name)
            if not config:
                raise ValueError(f"No configuration found for LLM handler: {handler_name}")
            
            self._llm_handlers[handler_name] = LLMHandler(**config)
        return self._llm_handlers[handler_name]

    def _get_qa_model(self):
        """Lazy initialization of QA model"""
        if self._qa_model is None:
            # Try to get embedding handler, but don't fail if not available
            embedding_handler = None
            try:
                embedding_handler = self._get_llm_handler("embedding")
            except:
                print("Warning: Embedding handler not available, embedding fallback disabled")
            
            self._qa_model = QAHandler(
                llm_handler=self._get_llm_handler("factual_qa"),
                kg_handler=self._get_kg_handler(),
                embedding_handler=embedding_handler,
                dataset_path=self.dataset_path,
                embeddings_path=self.embeddings_path
            )
        return self._qa_model

    def _get_recommender(self):
        """Lazy initialization of recommender"""
        if self._recommender is None:
            self._recommender = MovieRecommender(
                kg_handler=self._get_kg_handler(),
                dataset_path=self.dataset_path,
                embeddings_path=self.embeddings_path
            )
        return self._recommender

    def _get_multimedia(self):
        """Lazy initialization of multimedia handler"""
        if self._multimedia is None:
            self._multimedia = MultimediaHandler(dataset_path=self.dataset_path)
        return self._multimedia

    def get_answer(self, message: str, mode: int = 5):
        """
        Process a message with the specified mode.
        Modes: 1 (SPARQL), 2 (QA/embedding), 3 (recommendation), 4 (multimedia), 5 (auto)
        """
        if mode not in [1, 2, 3, 4, 5]:
            raise ValueError("Mode must be one of: 1 (SPARQL), 2 (QA/embedding), 3 (recommendation), 4 (multimedia), 5 (auto)")

        if mode == 1:
            return self._get_kg_handler().sparql_query(message)
        elif mode == 2:
            if "factual" in message.lower():
                submode = "factual"
            elif "embedding" in message.lower():
                submode = "embedding"
            else:
                submode = "factual"
            return self._get_qa_model().answer(message, submode=submode)
        elif mode == 3:
            return self._get_recommender().recommend(message)
        elif mode == 4:
            return self._get_multimedia().get_image(message)
        elif mode == 5:
            return self._handle_all(message)

    def _handle_all(self, message: str):
        """Auto-detect the appropriate mode based on message content"""
        # Check if the message is a factual question or an embedding question (skip llm intent)
        if "factual" in message.lower():
            return self._get_qa_model().answer(message, submode="factual")
        elif "embedding" in message.lower():
            return self._get_qa_model().answer(message, submode="embedding")

        # Use LLM to classify intent
        response = self.intent_classifier.generate_json_response(self.prompt_manager.get_prompt("intent_classifier", user_message=message))
        
        intent = "unknown"
        if response['success'] and response['json']:
            intent = response['json'].get("intent", "unknown")

        print(f"Detected intent: {intent}")

        # Route to the appropriate handler
        if intent == "sparql_query":
            print("SPARQL query detected")
            return self._get_kg_handler().sparql_query(message)
        elif intent == "recommendation_question":
            print("Recommendation question detected")
            return self._get_recommender().recommend(message)
        elif intent == "multimedia_question":
            print("Multimedia question detected")
            return self._get_multimedia().get_image(message)
        elif intent == "factual_question":
            print("Factual question detected")
            return self._get_qa_model().answer(message, submode="factual")
        elif intent == "greeting":
            print("Greeting detected")
            return "Hello I am GrayBarkingDog! How can I help you today?"
        else: # Fallback for 'unknown'
            print(f"Unknown intent detected, falling back to keyword matching...")
            low = message.strip().lower()
            if low.startswith("select") or low.startswith("prefix") or "where {" in low:
                print("SPARQL query detected by fallback")
                return self._get_kg_handler().sparql_query(message)
            if "recommend" in low or "i like" in low:
                print("Recommendation question detected by fallback")
                return self._get_recommender().recommend(message)
            if "picture" in low or "image" in low or "look like" in low:
                print("Multimedia question detected by fallback")
                return self._get_multimedia().get_image(message)
            print("Factual question detected by fallback")
            return self._get_qa_model().answer(message, submode="factual")
