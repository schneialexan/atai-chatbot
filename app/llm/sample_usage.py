import os
import time
from llm_handler import LLMHandler
from prompt_manager import PromptManager

# Instantiate a handler for a small, CPU-friendly text model.
# https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
# This model will be downloaded and loaded on first use.
# auto_load=True (default) means the model loads during initialization
llm = LLMHandler(
    model_repo="Qwen/Qwen3-0.6B-GGUF",
    model_file="Qwen3-0.6B-Q8_0.gguf",
    n_ctx=4096,  # Match training context for stability
    auto_load=True  # Model loads during initialization
)

def run_text_example(prompt_manager):
    """Demonstrates a simple text generation task."""
    print("=" * 60)
    print("🔤 TEXT GENERATION EXAMPLE")
    print("=" * 60)
    try:
        # Get a prompt from the manager
        prompt = prompt_manager.get_prompt("intent_classifier", user_message="Hello")
        
        print("\n📝 FORMATTED PROMPT:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)

        # Generate a response with better parameters for structured output
        response = llm.generate_response(
            prompt,
            max_tokens=1000,
            temperature=1.0,
            top_p=0.95,
            remove_thinking=True  # Remove thinking tags from reasoning models
        )
        print("\n🤖 LLM RESPONSE:")
        print("-" * 40)
        print(response['content'])

    except Exception as e:
        print(f"\n❌ TEXT EXAMPLE FAILED: {e}")
    print("=" * 60)
    print()

def run_json_extraction_example(prompt_manager):
    """Demonstrates JSON extraction from LLM responses."""
    print("=" * 60)
    print("🔍 JSON EXTRACTION EXAMPLE")
    print("=" * 60)
    try:
        # Get a prompt from the manager
        prompt = prompt_manager.get_prompt("intent_classifier", user_message="I like the film Terminator, can you recommend something similar?")
        
        print("\n📝 FORMATTED PROMPT:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)

        # Generate a response with JSON extraction
        json_response = llm.generate_json_response(
            prompt, 
            max_tokens=1000,
            temperature=0.1,
            top_p=0.7,
            remove_thinking=True  # Remove thinking tags from reasoning models
        )
        
        print("\n🤖 RAW LLM RESPONSE:")
        print("-" * 40)
        print(json_response['content'])
        print("-" * 40)
        
        print("\n📊 EXTRACTED JSON:")
        print("-" * 40)
        print(json_response['json'])
        print("-" * 40)
        
        if json_response['success']:
            print(f"\n✅ JSON EXTRACTION SUCCESS!")
        else:
            print(f"\n⚠️ JSON EXTRACTION FAILED!")
            print("   The LLM response may not contain valid JSON format")

    except Exception as e:
        print(f"\n❌ JSON EXTRACTION EXAMPLE FAILED: {e}")
    print("=" * 60)
    print()


def run_embedding_example():
    """Demonstrates a simple embedding task."""
    print("=" * 60)
    print("🧮 EMBEDDING EXAMPLE")
    print("=" * 60)
    try:
        # Instantiate a handler for a standard embedding model.
        # This model is downloaded automatically by the library.
        embedder = LLMHandler(
            model_repo="all-MiniLM-L6-v2", 
            model_type="embedding",
            auto_load=True
        )

        sentences = ['This is an example sentence', 'Each sentence is converted to a vector']
        
        print("\n📝 INPUT SENTENCES:")
        print("-" * 40)
        for i, sentence in enumerate(sentences, 1):
            print(f"{i}. {sentence}")
        print("-" * 40)
        
        # Generate embeddings
        embeddings = embedder.generate_embedding(sentences)
        
        print(f"\n📊 EMBEDDING RESULTS:")
        print("-" * 40)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📐 Shape of first embedding: {embeddings[0].shape}")
        print(f"🔢 Embedding dimensions: {embeddings[0].shape[0]}")

    except Exception as e:
        print(f"\n❌ EMBEDDING EXAMPLE FAILED: {e}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    print("🚀 STARTING LLM FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    print()
    
    try:
        prompt_manager = PromptManager()
        run_text_example(prompt_manager)
        run_json_extraction_example(prompt_manager)
        run_embedding_example()
        
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 60)
