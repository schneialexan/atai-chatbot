# Configuration file for the ATAI Chatbot
# This file contains the essential configurable settings for the chatbot

# Agent Configuration
AGENT_CONFIG = {
    "mode": 1,  # Default mode: 1=SPARQL, 2=QA, 3=Recommendation, 4=Multimedia, 5=Auto
    "dataset_path": "dataset/store/graph_cache.pkl",  # Path to the RDF dataset
    "speakeasy_host": "https://speakeasy.ifi.uzh.ch",  # Speakeasy server URL
    "preload_strategy": "mode_specific",  # Options: "all", "mode_specific", "none"
}
