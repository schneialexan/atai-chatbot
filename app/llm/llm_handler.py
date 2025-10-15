import os
import base64
from huggingface_hub import hf_hub_download
from json_parser import JSONParser

# Conditional imports
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
except ImportError:
    Llama, LlamaChatCompletionHandler = None, None, None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

class LLMHandler:
    """A handler for a single LLM, multimodal, or embedding model."""

    def __init__(self, model_repo: str, model_file: str = None, model_type: str = 'llm', 
                 mmproj_repo: str = None, mmproj_file: str = None, model_dir: str = "models",
                 n_gpu_layers: int = 0, n_ctx: int = 2048, auto_load: bool = True):
        print(f"Initializing LLMHandler for model: {model_repo}/{model_file}")
        self.model_repo = model_repo
        self.model_file = model_file
        self.model_type = model_type
        self.mmproj_repo = mmproj_repo or self.model_repo
        self.mmproj_file = mmproj_file
        self.model_dir = model_dir
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.model_path = None
        self.mmproj_path = None
        self.model = None
        self.json_parser = JSONParser()

        if model_type in ['llm', 'multimodal'] and Llama is None:
            raise ImportError("llama-cpp-python is not installed.")
        if model_type == 'embedding' and SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        
        # Load model during initialization if auto_load is True
        if auto_load:
            self._load_model()
    
    def __del__(self):
        """Destructor to properly unload the model when the object is destroyed."""
        self.unload_model()

    def _download_file(self, repo, filename):
        if not filename:
            return None
        
        os.makedirs(self.model_dir, exist_ok=True)
        file_path = os.path.join(self.model_dir, filename)

        if not os.path.exists(file_path):
            print(f"File not found. Downloading '{filename}' from '{repo}'...")
            try:
                hf_hub_download(repo_id=repo, filename=filename, local_dir=self.model_dir)
                print(f"Download complete for file: {filename}")
            except Exception as e:
                raise IOError(f"Failed to download file. Error: {e}")
        return file_path

    def _load_model(self):
        print(f"Loading '{self.model_type}' model: {self.model_repo}/{self.model_file or ''}...")
        if self.model_type in ['llm', 'multimodal']:
            self.model_path = self._download_file(self.model_repo, self.model_file)
            chat_handler = None
            if self.model_type == 'multimodal':
                if not self.mmproj_file:
                    raise ValueError("'mmproj_file' is required for multimodal models.")
                self.mmproj_path = self._download_file(self.mmproj_repo, self.mmproj_file)
                chat_handler = LlamaChatCompletionHandler.from_gguf_file(self.mmproj_path)

            self.model = Llama(
                model_path=self.model_path, 
                chat_handler=chat_handler, 
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False
            )
        
        elif self.model_type == 'embedding':
            self.model = SentenceTransformer(self.model_repo)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        print(f"Model: {self.model_repo}/{self.model_file} loaded successfully.")
    
    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is not None:
            print(f"Unloading model: {self.model_repo}/{self.model_file}")
            # For llama-cpp models, we can't explicitly unload, but we can set to None
            # The garbage collector will handle the cleanup
            self.model = None
            print("Model unloaded successfully.")
    
    def load_model(self):
        """Manually load the model if it wasn't loaded during initialization."""
        if self.model is None:
            self._load_model()
    
    def is_loaded(self):
        """Check if the model is currently loaded."""
        return self.model is not None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically unload model."""
        self.unload_model()

    def _remove_thinking_tags(self, text: str) -> str:
        """
        Remove thinking tags and thinking text from reasoning model outputs.
        
        Args:
            text (str): The text content to clean
            
        Returns:
            str: Cleaned text with thinking tags removed
        """
        import re
        
        # Common thinking tag patterns
        thinking_patterns = [
            r'<think>.*?</think>',  # <think>...</think>
            r'<thinking>.*?</thinking>',  # <thinking>...</thinking>
            r'<thought>.*?</thought>',    # <thought>...</thought>
            r'<reasoning>.*?</reasoning>', # <reasoning>...</reasoning>
            r'<internal>.*?</internal>',  # <internal>...</internal>
            r'<scratch>.*?</scratch>',    # <scratch>...</scratch>
            r'<work>.*?</work>',          # <work>...</work>
            r'<process>.*?</process>',    # <process>...</process>
        ]
        
        cleaned_text = text
        for pattern in thinking_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Multiple newlines to double newlines
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def generate_response(self, prompt: str, images: list = None, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9, remove_thinking: bool = False, **kwargs) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first or set auto_load=True during initialization.")

        if self.model_type not in ['llm', 'multimodal']:
            raise TypeError(f"generate_response is not available for '{self.model_type}' models.")

        # For text-only models, use simple string content
        if self.model_type == 'llm':
            raw_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        # For multimodal models, use content parts
        elif self.model_type == 'multimodal':
            content_parts = [{"type": "text", "text": prompt}]
            if images:
                for image_path in images:
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image file not found at {image_path}")
                    b64_image = image_to_base64(image_path)
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}})
            
            raw_response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": content_parts}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
        
        # Extract the text content
        if 'choices' in raw_response and len(raw_response['choices']) > 0:
            content = raw_response['choices'][-1]['message']['content']
        else:
            content = ""
        
        # Remove thinking tags if requested
        if remove_thinking and content:
            content = self._remove_thinking_tags(content)
        
        return {
            'raw_response': raw_response,
            'content': content,
            'success': content != ""
        }

    def generate_embedding(self, text, **kwargs):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first or set auto_load=True during initialization.")
        if self.model_type != 'embedding':
            raise TypeError("generate_embedding is only available for 'embedding' models.")
        return self.model.encode(text, **kwargs)
    
    def generate_json_response(self, prompt: str, images: list = None, max_tokens: int = 1024, 
                              temperature: float = 0.7, top_p: float = 0.9, strict: bool = False, 
                              remove_thinking: bool = False, **kwargs) -> dict:
        """
        Generate a response and extract JSON from it.
        
        Args:
            prompt (str): The input prompt
            images (list, optional): List of image paths for multimodal models
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            strict (bool): Whether to use strict JSON parsing
            remove_thinking (bool): Whether to remove thinking tags from reasoning models
            **kwargs: Additional arguments for the model
            
        Returns:
            dict: Dictionary containing the raw response and extracted JSON
        """
        # Generate the response using the main generate_response method
        response = self.generate_response(prompt, images, max_tokens, temperature, top_p, remove_thinking, **kwargs)
        
        # Extract JSON from the content
        extracted_json = self.json_parser.extract_json(response['content'], strict=strict)
        
        return {
            'raw_response': response['raw_response'],
            'content': response['content'],
            'json': extracted_json,
            'success': extracted_json is not None
        }
    
    def generate_all_json_responses(self, prompt: str, images: list = None, max_tokens: int = 1024, 
                                   temperature: float = 0.7, top_p: float = 0.9, remove_thinking: bool = False, **kwargs) -> dict:
        """
        Generate a response and extract all JSON objects from it.
        
        Args:
            prompt (str): The input prompt
            images (list, optional): List of image paths for multimodal models
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            remove_thinking (bool): Whether to remove thinking tags from reasoning models
            **kwargs: Additional arguments for the model
            
        Returns:
            dict: Dictionary containing the raw response and all extracted JSON objects
        """
        # Generate the response using the main generate_response method
        response = self.generate_response(prompt, images, max_tokens, temperature, top_p, remove_thinking, **kwargs)
        
        # Extract all JSON objects from the content
        extracted_json_list = self.json_parser.extract_all_json(response['content'])
        
        return {
            'raw_response': response['raw_response'],
            'content': response['content'],
            'json_list': extracted_json_list,
            'json_count': len(extracted_json_list),
            'success': len(extracted_json_list) > 0
        }
