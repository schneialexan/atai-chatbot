# Configuration file for the ATAI Chatbot
# This file contains the essential configurable settings for the chatbot

# Agent Configuration
AGENT_CONFIG = {
    "mode": 5,  # Default mode: 1=SPARQL, 2=QA, 3=Recommendation, 4=Multimedia, 5=Auto
    "dataset_path": "dataset/store/graph_cache.pkl",  # Path to the RDF dataset
    "embeddings_path": "dataset/embeddings",  # Path to the embeddings
    "speakeasy_host": "https://speakeasy.ifi.uzh.ch",  # Speakeasy server URL
    "preload_strategy": "mode_specific",  # Options: "all", "mode_specific", "none"
}

# LLM Configuration
LLM_CONFIG = {
    "factual_qa": {
        "backend": "gguf",
        "model_type": "llm",
        "model_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "model_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "auto_load": True
    },
    "embedding": {
        "backend": "transformer",
        "model_type": "embedding",
        "model_repo": "all-MiniLM-L6-v2",
        "auto_load": True
    },
    "intent_classifier": {
        "backend": "gguf",
        "model_type": "llm",
        "model_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "model_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 0,
        "n_ctx": 2048,
        "auto_load": True
    }
}
