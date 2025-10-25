# Conversational Chatbot for ATAI 2025 @ UZH

A sophisticated conversational chatbot that can answer questions using a knowledge graph through multiple approaches: direct SPARQL queries, natural language question answering, movie recommendations, and multimedia retrieval.

## Overview

The chatbot supports four main modes of operation:

1. **SPARQL Mode**: Direct querying of the RDF knowledge graph
2. **QA Mode**: Natural language question answering with entity extraction and SPARQL generation
3. **Recommendation Mode**: Movie recommendations based on user preferences
4. **Multimedia Mode**: Image retrieval and multimedia content handling

The system uses a modular architecture with local LLM models, embedding-based entity matching, and intelligent fallback mechanisms for robust question answering.

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd atai-chatbot
    ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For GPU acceleration:**
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   ```
   
   **Install spaCy English model (required for NER):**
   ```bash
   python -m spacy download en_core_web_trf
   ```

3. **Set up environment:**
   Create a `.env` file with your Speakeasy credentials:
    ```
    SPEAKEASY_USERNAME=<your-username>
    SPEAKEASY_PASSWORD=<your-password>
    ```

4. **Configure the system:**
   Edit `config.py` to set your preferred mode and settings.

## Usage

**Run the chatbot:**
```bash
python main.py
```

**Available modes:**
- `1`: SPARQL queries
- `2`: Natural language QA
- `3`: Movie recommendations  
- `4`: Multimedia retrieval
- `5`: Auto-detect mode

**Example questions:**
- SPARQL: `SELECT * WHERE { ?s ?p ?o } LIMIT 5`
- QA: `Who directed The Godfather?`
- Recommendations: `Recommend movies like The Godfather`
- Multimedia: `Show me a picture of The Godfather`

## Key Components

### QA Handler
Sophisticated question answering with:
- **Entity Extraction**: NER-based entity identification with fuzzy matching fallback
- **Property Mapping**: Synonym-based property identification with fuzzy string matching  
- **SPARQL Generation**: Automatic query construction from natural language
- **Answer Formatting**: LLM-powered natural language response generation

### Local LLM Framework
Modular framework supporting:
- **GGUF Models**: Local LLM inference with llama-cpp-python
- **Transformer Models**: NER and embedding models via Hugging Face
- **JSON Parsing**: Robust extraction of structured data from LLM responses
- **Resource Management**: Automatic model loading/unloading with context managers

## Testing

**Run all tests:**
```bash
python -m unittest testing/test_app.py
```

**Run specific tests:**
```bash
python -m unittest testing.test_app.TestApp.test_sparql_queries
python -m unittest testing.test_app.TestApp.test_factual_questions
```

## Project Structure

```
app/
├── core.py              # Main application class
├── qa_handler.py        # Question answering with entity extraction
├── kg_handler.py        # Knowledge graph and SPARQL handling
├── recommender.py       # Movie recommendation system
├── multimedia_handler.py # Image and multimedia retrieval
└── llm/                 # Local LLM framework
    ├── llama_cpp_handler.py    # GGUF model support
    ├── transformer_handler.py  # Hugging Face models
    ├── json_parser.py          # JSON extraction
    └── prompt_manager.py       # Prompt management

dataset/                 # Knowledge graph and embeddings
models/                  # Local model cache
testing/                 # Test suite
config.py               # Configuration settings
main.py                 # Application entry point
```
