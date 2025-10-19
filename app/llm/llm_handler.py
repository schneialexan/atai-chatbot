from .llama_cpp_handler import LlamaCppHandler
from .transformer_handler import TransformerHandler

def LLMHandler(backend: str, **kwargs):
    if backend == "gguf":
        return LlamaCppHandler(**kwargs)
    elif backend == "transformer":
        return TransformerHandler(**kwargs)
    else:
        raise ValueError(f"Invalid backend: {backend}")