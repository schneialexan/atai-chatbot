# Conversational Chatbot for ATAI 2025 @ UZH

<div align="center">
  <img src="logo.png" alt="ATAI Chatbot Logo" width="200"/>
</div>

A sophisticated conversational chatbot that can answer questions using a knowledge graph through multiple approaches: direct SPARQL queries, natural language question answering, movie recommendations, and multimedia retrieval.

## Overview

The chatbot supports four main modes of operation:

1. **SPARQL Mode**: Direct querying of the RDF knowledge graph
2. **QA Mode**: Natural language question answering with entity extraction and answer generation
   - **Factual Submode**: Traditional QA with entity extraction, SPARQL query execution, and direct knowledge graph lookups
   - **Embedding Submode**: Embedding-based similarity search using entity and relation embeddings to find the most likely answer
3. **Recommendation Mode**: Hybrid movie recommendation system combining content-based (TF-IDF) and collaborative filtering approaches
   - **Content-Based (TF-IDF)**: Recommends movies similar in content (genres, directors, themes) with hybrid scoring using common traits
   - **Collaborative Filtering**: Recommends movies based on user rating patterns (movies liked by users with similar tastes)
4. **Multimedia Mode**: Image retrieval and multimedia content handling

The system uses a modular architecture with local LLM models, embedding-based entity matching, and intelligent fallback mechanisms for robust question answering.

## Project Structure

```
atai-chatbot/
├── app/
│   ├── core.py                    # Main application class
│   ├── qa_handler.py              # Question answering with entity extraction
│   ├── kg_handler.py              # Knowledge graph and SPARQL handling
│   ├── recommender.py             # Movie recommendation system
│   ├── multimedia_handler.py      # Image and multimedia retrieval
│   └── llm/                       # Local LLM framework
│       ├── llama_cpp_handler.py   # GGUF model support
│       ├── transformer_handler.py # Hugging Face models
│       ├── json_parser.py         # JSON extraction
│       └── prompt_manager.py      # Prompt management
│
├── dataset/                       # Knowledge graph and embeddings
├── models/                        # Local model cache
├── testing/                       # Test suite
├── config.py                      # Configuration settings
└── main.py                        # Application entry point
```


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

    To create the Pickled graph binary use the `dataset/create_pickle_binary.py` file. This may take 2 minutes, since the whole graph has to be loaded.
## Usage

**Run the chatbot:**
```bash
python main.py
```

**Available modes:**
- `1`: SPARQL queries
- `2`: Natural language QA (with factual/embedding submode detection)
- `3`: Movie recommendations (hybrid content-based + collaborative filtering)
- `4`: Multimedia retrieval (WIP)
- `5`: Auto-detect mode (with submode detection)

**Example questions:**
- SPARQL: `SELECT * WHERE { ?s ?p ?o } LIMIT 5`
- QA Factual: `Who directed The Godfather?` or `Please answer this question with a factual approach: Who directed The Godfather?`
- QA Embedding: `Please answer this question with an embedding approach: Who directed The Godfather?`
- Recommendations: `I like The Godfather, can you recommend something similar?` or `Recommend movies like The Matrix and Inception`
- Multimedia: `Show me a picture of The Godfather`

## Key Components

### QA Handler
Sophisticated question answering with two distinct submodes:

#### Entity Extraction Pipeline (Shared)
Both submodes use the same sophisticated entity extraction pipeline:
- **Difficult Entity Extraction**: Pattern-based extraction for complex entities (quoted titles, colons, special characters)
- **NER Extraction**: spaCy-based named entity recognition
- **Entity Matching**: Fuzzy matching with longest-match strategy and word boundary handling
- **Property Mapping**: Synonym-based property identification with fuzzy string matching
- **Brute Force Fallback**: Comprehensive fallback when initial extraction fails

#### Factual Submode
- **Approach**: Direct knowledge graph querying via SPARQL
- **Process**: After entity and property extraction, executes SPARQL queries to retrieve exact answers
- **Use Case**: Questions requiring precise, factual answers from the knowledge graph
- **Output**: Direct extraction from RDF triples

#### Embedding Submode
- **Approach**: Embedding-based similarity search
- **Process**: 
  1. Retrieves entity embedding for the extracted entity
  2. Retrieves relation embedding for the identified property
  3. Computes `entity_embedding + relation_embedding` to form query embedding
  4. Finds the most similar entity using pairwise distance comparison
- **Use Case**: Questions where semantic similarity can help find related entities
- **Output**: Most likely answer entity based on embedding similarity

#### Answer Formatting
Both submodes use LLM-powered natural language formatting to generate human-readable responses from the raw results.

### Recommendation Handler
Sophisticated hybrid movie recommendation system that combines two complementary approaches:

#### Entity Extraction and Common Traits Inference
The recommendation pipeline starts with the same entity extraction used in QA:
- **Entity Extraction**: Uses NER and pattern matching to identify movie titles from user queries
- **Entity Resolution**: Maps movie titles to Wikidata URIs using fuzzy matching and film class filtering
- **Common Traits Inference**: Recursively explores the knowledge graph to find shared properties (directors, genres, actors, themes) across user's liked movies
- **Trait Aggregation**: Identifies the most common traits with source tracking and count-based ranking

#### Content-Based Recommendations (TF-IDF with Hybrid Scoring)
- **Approach**: TF-IDF vectorization of movie metadata (genres, directors, year, types, descriptions)
- **Initialization**: Metadata DataFrame and TF-IDF matrix are built during `MovieRecommender` initialization
- **Process**:
  1. Metadata DataFrame is built for all movies in the ratings dataset (with caching)
  2. TF-IDF matrix is created from combined movie features ("soup")
  3. Cosine similarity matrix is computed between all movies
  4. **Hybrid Scoring**: Boosts recommendations that match common traits identified from user's preferences
  5. Uses trait labels to create weighted query vectors for enhanced similarity matching
- **Use Case**: Finding movies with similar content, themes, or creative elements
- **Output**: Movies ranked by content similarity with trait-based boosting

#### Collaborative Filtering Recommendations
- **Approach**: Item-based collaborative filtering using user rating patterns
- **Initialization**: User-item matrix and item-item similarity matrix are built during `MovieRecommender` initialization
- **Process**:
  1. User-item rating matrix is built from ratings dataset, with missing values (unrated items) filled using average item ratings from `item_ratings.csv` instead of zeros. If an item is not found in `item_ratings.csv`, the global mean rating across all items is used as a fallback
  2. Item-item similarity is computed based on how users rate movies
  3. Finds movies similar to user's liked movies based on rating patterns
  4. Recommends movies that users with similar tastes have rated highly
- **Use Case**: Discovering movies through collective user preferences and rating patterns
- **Output**: Movies ranked by collaborative similarity scores

#### Recommendation Formatting
- **LLM-Powered Formatting**: Uses dedicated prompt to format both recommendation types into natural language
- **Structured Output**: Presents content-based and collaborative filtering recommendations separately with clear explanations
- **Fallback**: Simple text formatting if LLM is unavailable
- **Response Format**: Explains the difference between the two recommendation types and lists movies from both approaches

#### Initialization
- **Model Building**: Metadata DataFrame, TF-IDF matrix, and collaborative filtering models are built during `MovieRecommender` object initialization
- **Caching**: Metadata fetching uses caching for performance - first initialization may take time, but cached data speeds up subsequent initializations
- **Performance**: All models are ready immediately after initialization, making recommendation calls fast and consistent

### Local LLM Framework
Modular framework supporting:
- **GGUF Models**: Local LLM inference with llama-cpp-python
- **Transformer Models**: NER and embedding models via Hugging Face
- **JSON Parsing**: Robust extraction of structured data from LLM responses
- **Resource Management**: Automatic model loading/unloading with context managers

## Testing

### Running Factual Tests
Test factual question answering with different test cases covering directors, actors, genres, and release dates:

```bash
python -m unittest testing.test_factual_embeddings
```

### Running SPARQL Tests
Test direct SPARQL query functionality with different evaluation queries:

```bash
python -m unittest testing.test_sparql
```

## Architecture Diagrams

### High-Level Chat Loop Architecture (main.py)

```mermaid
graph TD
    A[User Message] --> B[Speakeasy Framework]
    B --> C[Agent.on_new_message]
    C --> D[App.get_answer]
    D --> E[Mode Detection]
    E --> F[Handler Selection]
    F --> G[Response Generation]
    G --> H[Speakeasy.post_messages]
    H --> I[User Response]
    
    J[User Reaction] --> K[Speakeasy Framework]
    K --> L[Agent.on_new_reaction]
    L --> M[Reaction Processing]
    M --> N[Reaction Response]
    N --> O[Speakeasy.post_messages]
    O --> P[User Response]
    
    Q[Session Management] --> C
    R[Event Logging] --> C
    S[Session Management] --> L
    T[Event Logging] --> L
    
    style A fill:#e1f5fe,color:#000000
    style J fill:#e1f5fe,color:#000000
    style C fill:#e8f5e8,color:#000000
    style D fill:#e8f5e8,color:#000000
    style E fill:#fff3e0,color:#000000
    style F fill:#fce4ec,color:#000000
    style L fill:#e8f5e8,color:#000000
    style M fill:#fff3e0,color:#000000
```

### High-Level App Core Architecture (core.py)

```mermaid
graph TD
    A[App Initialization] --> B[Load Configuration]
    B --> C[Initialize Prompt Manager]
    C --> D[Setup Lazy Loading]
    D --> E[Preload Strategy]
    E --> F[Intent Classifier]
    
    G[User Question] --> H[App.get_answer]
    H --> I[Mode Check]
    I --> J[Mode 1: SPARQL]
    I --> K[Mode 2: QA]
    I --> L[Mode 3: Recommendation]
    I --> M[Mode 4: Multimedia]
    I --> N[Mode 5: Auto]
    
    K --> K1[Submode Detection]
    K1 -->|factual| K2[Factual Submode]
    K1 -->|embedding| K3[Embedding Submode]
    K1 -->|default| K2
    
    N --> O[Intent Classification]
    O --> P[Intent Detection]
    P --> Q[Route to Handler]
    Q --> R[Handler Response]
    
    J --> S[KG Handler]
    K2 --> T[QA Handler]
    K3 --> T
    L --> U[Recommender]
    M --> V[Multimedia Handler]
    
    S --> W[Response]
    T --> W
    U --> W
    V --> W
    R --> W
    
    style A fill:#e1f5fe,color:#000000
    style G fill:#e1f5fe,color:#000000
    style H fill:#e8f5e8,color:#000000
    style I fill:#fff3e0,color:#000000
    style N fill:#fce4ec,color:#000000
    style O fill:#f3e5f5,color:#000000
```

### High-Level Knowledge Graph Handler (kg_handler.py)

```mermaid
graph TD
    A[KG Handler Initialization] --> B[Load RDF Dataset]
    B --> C[Parse Pickle File]
    C --> D[Initialize RDF Graph]
    
    E[SPARQL Query] --> F[sparql_query]
    F --> G[Query Validation]
    G --> H[Execute Query]
    H --> I[Parse Results]
    I --> J[Format Response]
    
    K[Entity Lookup] --> L[get_label_for_uri]
    L --> M[Local Graph Search]
    M --> O[Label Response]
    
    P[Property Discovery] --> Q[get_entity_property_labels]
    Q --> R[Query Entity Properties]
    R --> S[Return Property List]
    
    T[Metadata Extraction] --> U[get_entity_metadata_local]
    U --> V[Extract Types & Descriptions]
    V --> W[Return Metadata]
    
    style A fill:#e1f5fe,color:#000000
    style E fill:#e1f5fe,color:#000000
    style K fill:#e1f5fe,color:#000000
    style P fill:#e1f5fe,color:#000000
    style T fill:#e1f5fe,color:#000000
    style F fill:#e8f5e8,color:#000000
    style L fill:#e8f5e8,color:#000000
    style Q fill:#e8f5e8,color:#000000
    style U fill:#e8f5e8,color:#000000
```

### QA Handler Pipeline (qa_handler.py)

```mermaid
flowchart LR
    C["Extract Entities"] --> D["NER Extraction"] & E["Difficult Entities Pattern Matching"]
    E --> I["Collect Potential Entity-Property Candidates"]
    I --> J{"Candidates Found?"}
    J -- No --> K@{ label: "<span style=\"background-color:\">Brute Force Entity and Property Extraction Fallback</span>" }
    J -- Yes --> L["Select Best Entity"]
    P{"Candidates Found?"} -- No --> Q["No Entities Error"]
    P -- Yes --> L
    L --> R{"Submode?"}
    R -- Factual --> S["Factual: Execute SPARQL Query"]
    R -- Embedding --> W["Compute Query Embedding<br>entity + relation"]
    S --> U["Retrieve Answer from KG"]
    W --> X["Find Most Similar Entity<br>Pairwise Distance"]
    U --> Y["Format Answer with LLM"]
    X --> Y
    Y --> Z["Natural Language Response"]
    K --> P
    A["User Question"] --> C
    D --> I

    K@{ shape: rect}
    style C fill:#fff3e0,color:#000000
    style D fill:#fff3e0,color:#000000
    style E fill:#fff3e0,color:#000000
    style I fill:#f3e5f5,color:#000000
    style J fill:#fce4ec,color:#000000
    style K fill:#ffebee,color:#000000
    style L fill:#f3e5f5,color:#000000
    style Q fill:#ffcdd2,color:#000000
    style R fill:#e1bee7,color:#000000
    style S fill:#c8e6c9,color:#000000
    style W fill:#c8e6c9,color:#000000
    style U fill:#c8e6c9,color:#000000
    style X fill:#c8e6c9,color:#000000
    style Y fill:#e8f5e8,color:#000000
    style Z fill:#b2dfdb,color:#000000
    style A fill:#e1f5fe,color:#000000
```

### LLM Usage in Factual QA (qa_handler.py)

```mermaid
graph LR
    A["SPARQL Query Results<br>(Raw Data)"] --> B["Prepare Raw Data<br>(answers, metadata)"]
    B --> C["PromptManager.get_prompt<br>('result_to_natural_language')"]
    C --> D["LLM.generate_response<br>(formatted prompt)"]
    D --> E["Natural Language Answer"]
    
    style A fill:#fff3e0,color:#000000
    style B fill:#f3e5f5,color:#000000
    style C fill:#e1bee7,color:#000000
    style D fill:#c8e6c9,color:#000000
    style E fill:#b2dfdb,color:#000000
```

### LLM Usage in Intent Classification (core.py)

```mermaid
graph LR
    A["User Message"] --> B["PromptManager.get_prompt<br>('intent_classifier')"]
    B --> C["LLM.generate_json_response<br>(formatted prompt)"]
    C --> D["Extract Intent<br>(sparql_query, factual_question, etc.)"]
    D --> E["Route to Handler<br>(KG, QA, Recommender, Multimedia)"]
    
    style A fill:#e1f5fe,color:#000000
    style B fill:#e1bee7,color:#000000
    style C fill:#c8e6c9,color:#000000
    style D fill:#fff3e0,color:#000000
    style E fill:#f3e5f5,color:#000000
```

### Recommendation Handler Pipeline (recommender.py)

```mermaid
flowchart LR
    A["User Query"] --> B["Entity Extraction<br>& Resolution"]
    B --> C["Infer Common Traits<br>(KG exploration)"]
    
    C --> D1["Lookup Movie URIs<br>in Metadata DF"]
    D1 --> D2["Extract Features<br>(genres, directors,<br>year, types, desc)"]
    D2 --> D3["Compute Cosine<br>Similarity"]
    C --> D4["Build Trait Query<br>Vector"]
    D4 --> D5["Trait-Based<br>Score Boosting"]
    D3 --> D5
    D5 --> D["TF-IDF Recommendations"]
    
    C --> E["CF Recommendations<br>(Rating Patterns)"]
    D --> F["Format with LLM"]
    E --> F
    F --> G["Natural Language<br>Response"]
    
    style A fill:#e1f5fe,color:#000000
    style B fill:#fff3e0,color:#000000
    style C fill:#f3e5f5,color:#000000
    style D1 fill:#e1bee7,color:#000000
    style D2 fill:#e1bee7,color:#000000
    style D3 fill:#e1bee7,color:#000000
    style D4 fill:#e1bee7,color:#000000
    style D5 fill:#e1bee7,color:#000000
    style D fill:#e1bee7,color:#000000
    style E fill:#e1bee7,color:#000000
    style F fill:#e8f5e8,color:#000000
    style G fill:#b2dfdb,color:#000000
```

### LLM Usage in Recommendations (recommender.py)

```mermaid
graph LR
    A["TF-IDF Recommendations<br>(DataFrame)"] --> C["Prepare Structured Data<br>(JSON format)"]
    B["CF Recommendations<br>(DataFrame)"] --> C
    D["Common Traits<br>(List)"] --> C
    E["User Query"] --> C
    
    C --> F["PromptManager.get_prompt<br>('recommendation_response_formatter')"]
    F --> G["LLM.generate_response<br>(formatted prompt)"]
    G --> H["Natural Language Response<br>(Both recommendation types explained)"]
    
    style A fill:#e1bee7,color:#000000
    style B fill:#e1bee7,color:#000000
    style D fill:#f3e5f5,color:#000000
    style E fill:#e1f5fe,color:#000000
    style C fill:#f3e5f5,color:#000000
    style F fill:#e1bee7,color:#000000
    style G fill:#c8e6c9,color:#000000
    style H fill:#b2dfdb,color:#000000
```

---