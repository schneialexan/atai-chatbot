# Conversational Chatbot for ATAI 2025 @ UZH

This is a conversational chatbot that is designed to handle different tasks.

## Description

The project is divided into three intermediate evaluations and a final evaluation.

### Intermediate Evaluations

1.  **Answering Simple SPARQL Queries:** The agent needs to be able to answer plain SPARQL queries using the provided knowledge graph.
2.  **Answering Factual and Embedding Questions:** The agent needs to be able to answer real-world natural language questions by interpreting them, transforming them into executable SPARQL queries, and fetching the correct answers from the provided knowledge graph.
3.  **Answering Recommendation Questions:** The agent needs to be able to answer recommendation questions using the provided dataset.

### Final Evaluation

For the final project evaluation, the agent needs to be able to answer all four types of questions:

1.  **Factual Questions:** Answered based on the provided knowledge graph.
2.  **Embedding Questions:** Answered using the provided pre-trained embeddings.
3.  **Multimedia Questions:** Answered using the multi-media data set provided.
4.  **Recommendation Questions:** Answered based on the knowledge graph provided.

During the evaluation events, the chatbot will be tested on the Speakeasy platform.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd atai-chatbot
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.13 installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    
    **For GPU acceleration with NVIDIA GPUs:**
    A special installation of `llama-cpp-python` is required. Make sure you have the CUDA Toolkit installed on your system, then run the following command to reinstall it with CUDA support:
    ```bash
    CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```
    
    **For Transformer models:**
    The project uses Hugging Face transformers for NER and embedding models. These are automatically installed with the requirements, but you may need additional dependencies for specific models:
    ```bash
    pip install sentence-transformers  # For embedding models
    pip install transformers  # For NER models
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your Speakeasy credentials:
    ```
    SPEAKEASY_USERNAME=<your-username>
    SPEAKEASY_PASSWORD=<your-password>
    ```

4.  **Configure the chatbot:**
    Edit the `config.py` file to set your preferred mode and other settings:
    ```python
    AGENT_CONFIG = {
        "mode": 1,  # 1=SPARQL, 2=QA, 3=Recommendation, 4=Multimedia, 5=Auto
        "dataset_path": "dataset/graph.nt",
        "speakeasy_host": "https://speakeasy.ifi.uzh.ch",
        "preload_strategy": "mode_specific",  # Options: "all", "mode_specific", "none"
    }
    ```

## Usage

To run the chatbot, execute the `main.py` file:

```bash
python main.py
```

The chatbot uses **flexible initialization** - you can choose between lazy loading (for testing) or preloading (for production) using the `preload_strategy` configuration. The chatbot can operate in different modes, which must be specified when creating the Agent. The available modes are:

-   `1`: **SPARQL Mode**: For directly querying the RDF dataset.
-   `2`: **QA/Embedding Mode**: For answering factual questions.
-   `3`: **Recommendation Mode**: For getting movie recommendations.
-   `4`: **Multimedia Mode**: For retrieving images.
-   `5`: **Auto Mode**: The chatbot automatically determines the appropriate handler based on the user's message.

### Setting the Mode

The mode can be configured in two ways:

#### Option 1: Using the config file (Recommended)
Edit `config.py` to set your default mode:
```python
AGENT_CONFIG = {
    "mode": 1,  # Change this to your preferred mode
    # ... other settings
}
```

Then run the agent without specifying a mode:
```python
demo_bot = Agent(username, password)  # Uses mode from config
```

#### Option 2: Override the config
You can still override the config mode when creating the Agent:
```python
# Override config mode
demo_bot = Agent(username, password, mode=2)  # Uses mode=2 instead of config
```

### Model Preloading

The chatbot supports different preloading strategies:

#### Production Usage (preload all models)
```python
from app.core import App

# Initialize with all models preloaded
app = App(dataset_path, preload_strategy="all")
# All models are loaded at startup - first message will be fast
```

#### Production Usage (preload mode-specific models)
```python
from app.core import App

# Initialize with mode-specific preloading
app = App(dataset_path, preload_strategy="mode_specific")
# Only models for the configured mode are preloaded
```

#### Testing Usage (lazy loading)
```python
from app.core import App

# Initialize with lazy loading (default for testing)
app = App(dataset_path, preload_strategy="none")
# Models are loaded only when needed
```

#### Manual Mode-Specific Preloading
```python
from app.core import App

# Initialize without preloading
app = App(dataset_path, preload_strategy="none")

# Manually preload only the models needed for SPARQL queries
app.preload_models_for_mode(1)
```

### Programmatic Usage

You can also use the App class directly for more control:

```python
from app.core import App

# Create app instance (lazy loading by default)
app = App(dataset_path, preload_strategy="none")

# Or with preloading for production
app = App(dataset_path, preload_strategy="all")

# Use specific modes
result1 = app.get_answer("SELECT * WHERE {...}", mode=1)  # SPARQL
result2 = app.get_answer("Who directed this movie?", mode=2)  # QA
result3 = app.get_answer("Recommend movies like...", mode=3)  # Recommender
result4 = app.get_answer("Show me a picture of...", mode=4)  # Multimedia

# Auto-detect mode (default)
result5 = app.get_answer("Any question")  # Uses mode=5 (auto)
```

### Examples

-   **SPARQL Mode**:
    ```
    select * where { ?s ?p ?o } limit 5
    ```

-   **QA/Embedding Mode**:
    ```
    Who is the director of the movie "The Godfather"?
    ```

-   **Recommendation Mode**:
    ```
    I like the movie "The Godfather", can you recommend something similar?
    ```

-   **Multimedia Mode**:
    ```
    Show me a picture of "The Godfather" poster.
    ```

## Local LLM Framework

The project includes a modular framework for running open-source language models locally, located in the `app/llm/` directory. This framework is designed for flexibility and performance, allowing you to integrate various types of models into the chatbot pipeline.

### Features

-   **Run Models Locally:** No external APIs or servers are needed.
-   **Modular Architecture:** Factory pattern with specialized handlers for different model types (GGUF and Transformer models).
-   **Multiple Model Types:** Natively supports standard LLMs, multimodal models (text + image), NER models, and sentence-embedding models.
-   **Automatic Downloading:** Models are automatically downloaded from Hugging Face on first use and cached locally in the `models/` directory.
-   **Efficient Model Loading:** Models load once during initialization and stay loaded for optimal performance.
-   **Flexible Loading Options:** Choose between eager loading (default) or lazy loading for different use cases.
-   **Automatic Resource Management:** Built-in cleanup with destructors and context manager support.
-   **Configurable Performance:** Easily configure GPU offloading (`n_gpu_layers`) and context window size (`n_ctx`) for each model instance.
-   **Decoupled Prompt Management:** Prompts are managed in a simple `prompts.json` file, separate from the application logic.
-   **JSON Parser Integration:** Built-in JSON extraction from LLM responses with robust error handling.
-   **Reasoning Model Support:** Optional thinking tag removal for reasoning models (removes `<thinking>`, `<thought>`, `<reasoning>`, etc. tags).

### Core Components

-   `llm_handler.py`: Factory function that creates appropriate handlers based on backend type (GGUF or Transformer).
-   `llama_cpp_handler.py`: Specialized handler for GGUF models using llama-cpp-python, supporting both text and multimodal models.
-   `transformer_handler.py`: Specialized handler for Hugging Face transformer models, supporting NER and embedding models.
-   `json_parser.py`: A robust JSON parser that can extract structured data from LLM responses, handling mixed text, code blocks, and malformed JSON gracefully.
-   `prompt_manager.py`: Contains the `PromptManager` class, which loads templates from `prompts.json` and formats them with dynamic data.

### Basic Usage Example

You can instantiate multiple handlers for different models and use them as needed.

#### Sequential Usage

```python
from app.llm.llm_handler import LLMHandler

# 1. Create a handler for a text-based LLM (GGUF model)
# Model loads automatically during initialization (efficient!)
llm = LLMHandler(
    backend="gguf",
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf",
    n_gpu_layers=0,  # CPU-only for this example
    n_ctx=4096,
    auto_load=True  # Default: loads model during initialization
)

# 2. Create a handler for an embedding model (Transformer model)
embedder = LLMHandler(
    backend="transformer",
    model_repo="all-MiniLM-L6-v2", 
    model_type="embedding"
)

# 3. Create a handler for NER (Named Entity Recognition)
ner_handler = LLMHandler(
    backend="transformer",
    model_repo="dslim/distilbert-NER",
    model_type="ner"
)

# 4. Generate a text response (no loading delay!)
response = llm.generate_response("What is the capital of France?")
print(response['content'])

# 5. Generate sentence embeddings (no loading delay!)
embeddings = embedder.generate_embedding(["Hello world", "This is a test"])
print(len(embeddings))

# 6. Extract entities from text
ner_result = ner_handler.generate_ner_response("I live in New York City")
print(ner_result['content'])

# 7. Clean up resources when done
llm.unload_model()
embedder.unload_model()
ner_handler.unload_model()
```

#### Parallel Inference

Because each `LLMHandler` is a self-contained object, you can run multiple handlers at the same time using Python's concurrency tools. This is useful for running an intent classifier and a response generator simultaneously.

```python
import threading
from app.llm.llm_handler import LLMHandler

# 1. Create handlers for two different models
intent_classifier = LLMHandler(
    backend="gguf",
    model_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

embedding_model = LLMHandler(
    backend="transformer",
    model_repo="all-MiniLM-L6-v2", 
    model_type="embedding"
)

# 2. Define functions for each thread to execute
def run_classifier():
    print("Classifier started...")
    response = intent_classifier.generate_response("Classify this intent: 'Recommend a movie'")
    print("Classifier finished.")

def run_embedder():
    print("Embedder started...")
    embedding = embedding_model.generate_embedding(["This is a test sentence."])
    print("Embedder finished.")

# 3. Create and start the threads
classifier_thread = threading.Thread(target=run_classifier)
embedder_thread = threading.Thread(target=run_embedder)

classifier_thread.start()
embedder_thread.start()

# 4. Wait for both threads to complete
classifier_thread.join()
embedder_thread.join()

print("\nBoth models ran in parallel.")
```

### Model Lifecycle Management

The LLM framework provides efficient model lifecycle management with multiple loading strategies and automatic resource cleanup.

#### Eager Loading (Default)

Models load automatically during initialization for optimal performance:

```python
from app.llm.llm_handler import LLMHandler

# Model loads during initialization (default behavior)
llm = LLMHandler(
    backend="gguf",
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf",
    n_ctx=4096,
    auto_load=True  # Default: True
)

# Ready to use immediately - no loading delay
response = llm.generate_response("Hello, world!")
```

#### Lazy Loading

For scenarios where you want to control when models are loaded:

```python
# Create handler without loading the model
llm = LLMHandler(
    backend="gguf",
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf",
    auto_load=False  # Don't load during initialization
)

# Check if model is loaded
print(f"Model loaded: {llm.is_loaded()}")  # False

# Load model manually when needed
llm.load_model()
print(f"Model loaded: {llm.is_loaded()}")  # True

# Now ready to use
response = llm.generate_response("Hello, world!")
```

#### Context Manager Support

Automatic resource cleanup with context managers:

```python
# Model automatically unloaded when exiting context
with LLMHandler(
    backend="gguf",
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf"
) as llm:
    response = llm.generate_response("Hello, world!")
    print(response['content'])
# Model automatically unloaded here
```

#### Manual Resource Management

```python
llm = LLMHandler(backend="gguf", model_repo="...", model_file="...")

# Use the model
response = llm.generate_response("Hello!")

# Manually unload when done
llm.unload_model()

# Check status
print(f"Model loaded: {llm.is_loaded()}")  # False
```

#### Performance Benefits

| **Loading Strategy** | **Use Case** | **Performance** |
|---------------------|--------------|-----------------|
| **Eager Loading** | Production, repeated use | ⚡ Fastest - no loading delay |
| **Lazy Loading** | Conditional usage | 🔧 Flexible - load when needed |
| **Context Manager** | Temporary usage | 🛡️ Safe - automatic cleanup |

#### Best Practices

- **Production**: Use eager loading (`auto_load=True`) for consistent performance
- **Development**: Use lazy loading (`auto_load=False`) for faster startup
- **Temporary Usage**: Use context managers for automatic cleanup
- **Memory Management**: Always unload models when done to free memory

### JSON Parser Integration

The LLM framework includes a powerful JSON parser that can extract structured data from LLM responses. This is particularly useful for intent classification, SPARQL query generation, and other tasks that require structured output.

#### Basic JSON Extraction

```python
from app.llm.llm_handler import LLMHandler

# Create an LLM handler
llm = LLMHandler(
    backend="gguf",
    model_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# Generate response and extract JSON in one call
json_response = llm.generate_json_response("Classify this intent: 'Recommend a movie'")
print(json_response['json'])  # {'intent': 'recommendation_question', 'confidence': 0.95}
```

#### Advanced JSON Extraction

```python
# Extract JSON from existing text
text = 'Here is the result: {"intent": "factual_question", "confidence": 0.95}'
json_obj = llm.extract_json_from_text(text)
print(json_obj)  # {'intent': 'factual_question', 'confidence': 0.95}

# Extract multiple JSON objects
text_with_multiple = 'First: {"a": 1} and second: {"b": 2}'
all_json = llm.extract_all_json_from_text(text_with_multiple)
print(all_json)  # [{'a': 1}, {'b': 2}]
```

#### JSON Parser Features

The JSON parser handles various scenarios:

- **Mixed text**: Extracts JSON from text with other content
- **Code blocks**: Finds JSON in ```json``` code blocks
- **Malformed JSON**: Gracefully handles invalid JSON
- **Multiple objects**: Can extract all JSON objects from text
- **Real-world examples**: Works with complex LLM outputs

### Reasoning Model Support

The LLM framework includes built-in support for reasoning models that output their thinking process in special tags. You can automatically remove these thinking tags to get clean, final responses.

#### Supported Thinking Tags

The framework automatically removes the following thinking tag patterns:
- `<think>...</think>`
- `<thinking>...</thinking>`
- `<thought>...</thought>`
- `<reasoning>...</reasoning>`
- `<internal>...</internal>`
- `<scratch>...</scratch>`
- `<work>...</work>`
- `<process>...</process>`

#### Basic Usage

```python
from app.llm.llm_handler import LLMHandler

# Create an LLM handler for a reasoning model
llm = LLMHandler(
    backend="gguf",
    model_repo="microsoft/Phi-3.5-mini-instruct-GGUF",
    model_file="Phi-3.5-mini-instruct-q4.gguf"
)

# Generate response with thinking tags removed
response = llm.generate_response(
    "Solve this math problem: What is 15% of 200?",
    remove_thinking=True  # Remove thinking tags automatically
)
print(response['content'])  # Clean final answer without thinking process
```

#### JSON Extraction with Thinking Removal

```python
# Generate JSON response with thinking tags removed
json_response = llm.generate_json_response(
    "Classify this intent: 'I want to watch a comedy movie'",
    remove_thinking=True  # Remove thinking tags before JSON extraction
)

if json_response['success']:
    print(json_response['json'])  # Clean JSON without thinking tags
else:
    print("JSON extraction failed")
```

#### Raw vs Clean Responses

```python
# Get raw response with thinking tags
raw_response = llm.generate_response("Explain quantum computing")
print("Raw response:", raw_response['content'])

# Get clean response without thinking tags
clean_response = llm.generate_response("Explain quantum computing", remove_thinking=True)
print("Clean response:", clean_response['content'])
```

## QA Handler

The project includes a sophisticated Question Answering (QA) handler that can process natural language questions and convert them into SPARQL queries to retrieve answers from the knowledge graph.

### Features

-   **Entity Extraction:** Automatically identifies entities in user questions using multiple strategies (cached entities, explicit quotes, NER fallback).
-   **Property Identification:** Maps user questions to knowledge graph properties using fuzzy matching and synonym mapping.
-   **SPARQL Query Generation:** Constructs executable SPARQL queries based on extracted entities and properties.
-   **Natural Language Formatting:** Uses LLM to format raw query results into human-readable answers.
-   **Caching System:** Caches entities and properties for faster subsequent queries.
-   **Embedding Support:** Optional embedding-based entity matching for improved accuracy.
-   **Wikidata Integration:** Fallback to Wikidata API for entity labels not found in the local knowledge graph.

### QA Handler Components

-   **Entity Extraction Pipeline:**
    1. **Cached Entity Matching:** Fast lookup using pre-loaded entity cache
    2. **Explicit Quote Extraction:** Handles entities enclosed in single quotes
    3. **NER Fallback:** Uses transformer-based NER model for complex cases
    4. **Fuzzy Matching:** Handles typos and variations in entity names

-   **Property Identification:**
    1. **Synonym Mapping:** Maps user terms to canonical property names
    2. **Fuzzy String Matching:** Handles variations in property names
    3. **Context-Aware Processing:** Considers question context for better matching

-   **SPARQL Query Construction:**
    1. **Template-Based Generation:** Uses predefined SPARQL templates
    2. **Entity URI Resolution:** Maps entity names to knowledge graph URIs
    3. **Property URI Mapping:** Converts property names to SPARQL predicates

### Usage Example

```python
from app.qa_handler import QAHandler
from app.llm.llm_handler import LLMHandler
from app.sparql_handler import LocalSPARQL

# Initialize required handlers
llm_handler = LLMHandler(backend="gguf", model_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF", model_file="qwen2.5-0.5b-instruct-q4_k_m.gguf")
ner_handler = LLMHandler(backend="transformer", model_repo="dslim/distilbert-NER", model_type="ner")
sparql_handler = LocalSPARQL(dataset_path="dataset/store/graph_cache.pkl")

# Create QA handler
qa_handler = QAHandler(
    llm_handler=llm_handler,
    sparql_handler=sparql_handler,
    ner_handler=ner_handler,
    dataset_path="dataset",
    embeddings_path="dataset/store/embeddings"
)

# Process a question
question = "Who directed the movie The Godfather?"
answer = qa_handler.answer(question)
print(answer)  # "Based on my knowledge graph: Francis Ford Coppola"
```

### Configuration Files

The QA handler uses several configuration files:

-   **`cached_entities.json`:** Pre-computed entity cache for fast entity lookup
-   **`cached_properties.json`:** Pre-computed property cache for fast property lookup  
-   **`property_synonyms.json`:** Maps user terms to canonical property names
-   **`prompts.json`:** Contains templates for natural language formatting

### Advanced Features

#### Entity Extraction Strategies

```python
# The QA handler uses multiple strategies for entity extraction:

# 1. Cached entity matching (fastest)
entities = qa_handler._extract_entities("Tell me about The Godfather")

# 2. Explicit quote extraction
entities = qa_handler._extract_entities("Tell me about 'The Godfather' movie")

# 3. NER fallback for complex cases
entities = qa_handler._extract_entities("I want to know about Francis Ford Coppola")
```

#### Property Synonym Mapping

```python
# The QA handler supports synonym mapping for better property identification:

# User question: "What is the cast of The Godfather?"
# Mapped to property: "starring" (via synonym mapping)

# User question: "Who are the actors in The Godfather?"  
# Mapped to property: "starring" (via synonym mapping)
```

#### Embedding-Based Entity Matching

```python
# Optional embedding-based entity matching for improved accuracy
qa_handler = QAHandler(
    llm_handler=llm_handler,
    sparql_handler=sparql_handler,
    ner_handler=ner_handler,
    embedding_handler=embedding_handler,  # Optional embedding handler
    dataset_path="dataset",
    embeddings_path="dataset/store/embeddings"
)
```

## Testing

### Running All Tests

To run all automated tests, navigate to your project's root directory and run the following command:

```bash
python -m unittest testing/test_app.py
```

### Running Individual Tests

You can run specific test methods to test individual components:

```bash
# Test SPARQL functionality
python -m unittest testing.test_app.TestApp.test_sparql_queries

# Test QA functionality  
python -m unittest testing.test_app.TestApp.test_factual_questions
python -m unittest testing.test_app.TestApp.test_embedding_questions

# Test multimedia functionality
python -m unittest testing.test_app.TestApp.test_multimedia_questions

# Test recommendation functionality
python -m unittest testing.test_app.TestApp.test_recommendation_questions
```

## Project Structure

-   `app/`: Contains the core logic of the chatbot.
    -   `core.py`: The main application class with lazy initialization for different modes.
    -   `sparql_handler.py`: Handles SPARQL queries (loaded on-demand).
    -   `qa_handler.py`: Handles factual questions with entity extraction and SPARQL query generation.
    -   `recommender.py`: Handles movie recommendations (loaded on-demand).
    -   `multimedia_handler.py`: Handles multimedia requests (loaded on-demand).
    -   `llm/`: Contains the local language model framework.
        -   `llm_handler.py`: Factory function for creating appropriate model handlers.
        -   `llama_cpp_handler.py`: Specialized handler for GGUF models using llama-cpp-python.
        -   `transformer_handler.py`: Specialized handler for Hugging Face transformer models.
        -   `json_parser.py`: A robust JSON parser for extracting structured data from LLM responses.
        -   `prompt_manager.py`: Manages loading and formatting prompts.
        -   `prompts.json`: A file containing all prompt templates.
    -   `cached_entities.json`: Pre-computed entity cache for fast entity lookup.
    -   `cached_properties.json`: Pre-computed property cache for fast property lookup.
    -   `property_synonyms.json`: Maps user terms to canonical property names.
-   `dataset/`: Contains the dataset for the chatbot.
-   `models/`: Local cache for downloaded GGUF models. This directory is created automatically and is ignored by Git.
-   `speakeasypy/`: Contains the Speakeasy framework for chatbot communication.
-   `testing/`: Contains test files for the application.
-   `main.py`: The entry point of the application with configurable mode support.
-   `config.py`: Configuration file for agent settings and modes.
-   `requirements.txt`: The list of Python dependencies.
-   `.env`: The file to store your Speakeasy credentials.

## Configuration

The chatbot uses a centralized configuration system through `config.py`. This makes it easy to change settings without modifying the main code.

### Available Configuration Options

```python
# Agent Configuration
AGENT_CONFIG = {
    "mode": 2,  # Default mode: 1=SPARQL, 2=QA, 3=Recommendation, 4=Multimedia, 5=Auto
    "dataset_path": "dataset/store/graph_cache.pkl",  # Path to the RDF dataset
    "embeddings_path": "dataset/store/embeddings",  # Path to the embeddings
    "speakeasy_host": "https://speakeasy.ifi.uzh.ch",  # Speakeasy server URL
    "preload_strategy": "mode_specific",  # Options: "all", "mode_specific", "none"
}

# LLM Configuration
LLM_CONFIG = {
    "factual_qa": {
        "backend": "gguf",
        "model_type": "llm",
        "model_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "model_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "auto_load": True
    },
    "ner": {
        "backend": "transformer",
        "model_type": "ner",
        "model_repo": "dslim/distilbert-NER",
        "auto_load": True
    },
    "embedding": {
        "backend": "transformer",
        "model_type": "embedding",
        "model_repo": "all-MiniLM-L6-v2",
        "auto_load": True
    }
}
```

#### Preloading Configuration

- **`preload_strategy`**: 
  - `"all"`: Preload all models at startup (best for production with multiple modes)
  - `"mode_specific"`: Preload only models needed for the configured mode (best for production with single mode)
  - `"none"`: No preloading (lazy loading only - best for testing)
