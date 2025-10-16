import json
import re
from typing import Dict, List, Any, Optional, Union


class JSONParser:
    """
    A utility class for extracting and parsing JSON from LLM output text.
    Handles cases where JSON might be embedded within other text or have formatting issues.
    """
    
    def __init__(self):
        """Initialize the JSON parser."""
        pass
    
    def extract_json(self, text: str, strict: bool = False) -> Optional[Dict[str, Any]]:
        """
        Extract the first valid JSON object from text.
        
        Args:
            text (str): The text containing JSON
            strict (bool): If True, only return JSON if it's the only content. If False, extract JSON from mixed content.
            
        Returns:
            Optional[Dict[str, Any]]: The parsed JSON object, or None if no valid JSON found
        """
        if not text or not isinstance(text, str):
            return None
            
        # Try to parse the entire text as JSON first (strict mode)
        if strict:
            try:
                return json.loads(text.strip())
            except (json.JSONDecodeError, ValueError):
                return None
        
        # Look for JSON objects in the text using regex
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\{.*?\}',  # Any content between braces
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = self._clean_json_string(match)
                    if cleaned:
                        return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # Try to find JSON between code blocks or specific markers
        code_block_patterns = [
            r'```json\s*\n(.*?)\n```',  # JSON in code blocks
            r'```\s*\n(.*?)\n```',      # Any code blocks
            r'Your JSON Response:\s*\n(.*?)(?:\n\n|\n#|$)',  # After "Your JSON Response:"
            r'Response:\s*\n(.*?)(?:\n\n|\n#|$)',  # After "Response:"
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    cleaned = self._clean_json_string(match)
                    if cleaned:
                        return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return None
    
    def extract_all_json(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all valid JSON objects from text.
        
        Args:
            text (str): The text containing JSON objects
            
        Returns:
            List[Dict[str, Any]]: List of all parsed JSON objects found
        """
        if not text or not isinstance(text, str):
            return []
        
        json_objects = []
        
        # Try to find all JSON objects using regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                cleaned = self._clean_json_string(match)
                if cleaned:
                    json_obj = json.loads(cleaned)
                    json_objects.append(json_obj)
            except (json.JSONDecodeError, ValueError):
                continue
        
        return json_objects
    
    def _clean_json_string(self, json_str: str) -> Optional[str]:
        """
        Clean and prepare a JSON string for parsing.
        
        Args:
            json_str (str): Raw JSON string
            
        Returns:
            Optional[str]: Cleaned JSON string, or None if invalid
        """
        if not json_str:
            return None
        
        # Remove leading/trailing whitespace
        cleaned = json_str.strip()
        
        # Remove common prefixes/suffixes that might interfere
        prefixes_to_remove = [
            'Your JSON Response:',
            'Response:',
            'JSON:',
            '```json',
            '```',
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing code block markers
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        
        # Ensure it starts and ends with braces
        if not cleaned.startswith('{'):
            # Try to find the first opening brace
            start_idx = cleaned.find('{')
            if start_idx != -1:
                cleaned = cleaned[start_idx:]
            else:
                return None
        
        if not cleaned.endswith('}'):
            # Try to find the last closing brace
            end_idx = cleaned.rfind('}')
            if end_idx != -1:
                cleaned = cleaned[:end_idx + 1]
            else:
                return None
        
        return cleaned if cleaned else None
    
    def validate_json(self, json_obj: Any) -> bool:
        """
        Validate that an object is a proper JSON-serializable structure.
        
        Args:
            json_obj (Any): Object to validate
            
        Returns:
            bool: True if the object is JSON-serializable, False otherwise
        """
        try:
            json.dumps(json_obj)
            return True
        except (TypeError, ValueError):
            return False
    
    def format_json(self, json_obj: Any, indent: int = 2) -> str:
        """
        Format a JSON object as a pretty-printed string.
        
        Args:
            json_obj (Any): Object to format
            indent (int): Number of spaces for indentation
            
        Returns:
            str: Pretty-printed JSON string
        """
        try:
            return json.dumps(json_obj, indent=indent, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot format object as JSON: {e}")

