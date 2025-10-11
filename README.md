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

### Running Tests by Pattern

You can also run tests that match a specific pattern:

```bash
# Run all SPARQL-related tests
python -m unittest testing.test_app.TestApp -k "sparql"

# Run all QA-related tests
python -m unittest testing.test_app.TestApp -k "qa"

# Run all multimedia-related tests
python -m unittest testing.test_app.TestApp -k "multimedia"

# Run all recommendation-related tests
python -m unittest testing.test_app.TestApp -k "recommendation"
```

## Project Structure

-   `app/`: Contains the core logic of the chatbot.
    -   `core.py`: The main application class with lazy initialization for different modes.
    -   `sparql_handler.py`: Handles SPARQL queries (loaded on-demand).
    -   `qa_handler.py`: Handles factual questions (loaded on-demand).
    -   `recommender.py`: Handles movie recommendations (loaded on-demand).
    -   `multimedia_handler.py`: Handles multimedia requests (loaded on-demand).
    -   `llm/`: Contains the language model module related files.
-   `dataset/`: Contains the dataset for the chatbot.
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
