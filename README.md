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
    For GPU acceleration with NVIDIA GPUs, a special installation of `llama-cpp-python` is required. Make sure you have the CUDA Toolkit installed on your system, then run the following command to reinstall it with CUDA support:
    ```bash
    CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
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
-   **Unified Interface:** The `LLMHandler` class provides a consistent way to interact with different model types.
-   **Multiple Model Types:** Natively supports standard LLMs, multimodal models (text + image), and sentence-embedding models.
-   **Automatic Downloading:** Models are automatically downloaded from Hugging Face on first use and cached locally in the `models/` directory.
-   **Efficient Model Loading:** Models load once during initialization and stay loaded for optimal performance.
-   **Flexible Loading Options:** Choose between eager loading (default) or lazy loading for different use cases.
-   **Automatic Resource Management:** Built-in cleanup with destructors and context manager support.
-   **Configurable Performance:** Easily configure GPU offloading (`n_gpu_layers`) and context window size (`n_ctx`) for each model instance.
-   **Decoupled Prompt Management:** Prompts are managed in a simple `prompts.json` file, separate from the application logic.
-   **JSON Parser Integration:** Built-in JSON extraction from LLM responses with robust error handling.
-   **Reasoning Model Support:** Optional thinking tag removal for reasoning models (removes `<thinking>`, `<thought>`, `<reasoning>`, etc. tags).

### Core Components

-   `llm_handler.py`: Contains the `LLMHandler` class. Each instance of this class manages a single model, providing methods for text generation, embedding, multimodal inference, and JSON extraction.
-   `json_parser.py`: A robust JSON parser that can extract structured data from LLM responses, handling mixed text, code blocks, and malformed JSON gracefully.
-   `prompt_manager.py`: Contains the `PromptManager` class, which loads templates from `prompts.json` and formats them with dynamic data.
-   `sample_usage.py`: A runnable script that provides clear, practical examples of how to instantiate and use the `LLMHandler` for various tasks.

### Basic Usage Example

You can instantiate multiple handlers for different models and use them as needed.

#### Sequential Usage

```python
from app.llm.llm_handler import LLMHandler

# 1. Create a handler for a text-based LLM
# Model loads automatically during initialization (efficient!)
llm = LLMHandler(
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf",
    n_gpu_layers=0,  # CPU-only for this example
    n_ctx=4096,
    auto_load=True  # Default: loads model during initialization
)

# 2. Create a handler for an embedding model
embedder = LLMHandler(model_repo="all-MiniLM-L6-v2", model_type="embedding")

# 3. Generate a text response (no loading delay!)
response = llm.generate_response("What is the capital of France?")
print(response['content'])

# 4. Generate sentence embeddings (no loading delay!)
embeddings = embedder.generate_embedding(["Hello world", "This is a test"])
print(embeddings.shape)

# 5. Clean up resources when done
llm.unload_model()
embedder.unload_model()
```

#### Parallel Inference

Because each `LLMHandler` is a self-contained object, you can run multiple handlers at the same time using Python's concurrency tools. This is useful for running an intent classifier and a response generator simultaneously.

```python
import threading
from app.llm.llm_handler import LLMHandler

# 1. Create handlers for two different models
intent_classifier = LLMHandler(
    model_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

embedding_model = LLMHandler(model_repo="all-MiniLM-L6-v2", model_type="embedding")

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

To see more detailed examples, run the sample script:
```bash
python app/llm/sample_usage.py
```

### Model Lifecycle Management

The LLM framework provides efficient model lifecycle management with multiple loading strategies and automatic resource cleanup.

#### Eager Loading (Default)

Models load automatically during initialization for optimal performance:

```python
from app.llm.llm_handler import LLMHandler

# Model loads during initialization (default behavior)
llm = LLMHandler(
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
    model_repo="ibm-granite/granite-4.0-h-micro-GGUF",
    model_file="granite-4.0-h-micro-Q2_K.gguf"
) as llm:
    response = llm.generate_response("Hello, world!")
    print(response['content'])
# Model automatically unloaded here
```

#### Manual Resource Management

```python
llm = LLMHandler(model_repo="...", model_file="...")

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
    -   `qa_handler.py`: Handles factual questions (loaded on-demand).
    -   `recommender.py`: Handles movie recommendations (loaded on-demand).
    -   `multimedia_handler.py`: Handles multimedia requests (loaded on-demand).
    -   `llm/`: Contains the local language model framework.
        -   `llm_handler.py`: The core handler class for managing individual models with JSON extraction capabilities.
        -   `json_parser.py`: A robust JSON parser for extracting structured data from LLM responses.
        -   `prompt_manager.py`: Manages loading and formatting prompts.
        -   `prompts.json`: A file containing all prompt templates.
        -   `sample_usage.py`: A script demonstrating how to use the LLM framework.
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
AGENT_CONFIG = {
    "mode": 1,  # Default mode (1-5)
    "dataset_path": "dataset/graph.nt",  # Path to RDF dataset
    "speakeasy_host": "https://speakeasy.ifi.uzh.ch",  # Speakeasy server
    "preload_strategy": "mode_specific",  # Options: "all", "mode_specific", "none"
}
```

#### Preloading Configuration

- **`preload_strategy`**: 
  - `"all"`: Preload all models at startup (best for production with multiple modes)
  - `"mode_specific"`: Preload only models needed for the configured mode (best for production with single mode)
  - `"none"`: No preloading (lazy loading only - best for testing)
