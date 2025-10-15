import json
import os

class PromptManager:
    """Manages loading and formatting of prompts from a JSON file."""

    def __init__(self, prompts_file_path=None):
        """
        Initializes the PromptManager.

        Args:
            prompts_file_path (str, optional): The path to the prompts JSON file.
                                                 Defaults to 'prompts.json' in the same directory.
        """
        if prompts_file_path is None:
            prompts_file_path = os.path.join(os.path.dirname(__file__), 'prompts.json')

        if not os.path.exists(prompts_file_path):
            raise FileNotFoundError(f"Prompts file not found at: {prompts_file_path}")

        with open(prompts_file_path, 'r') as f:
            self.prompts = json.load(f)

    def get_prompt(self, task_name: str, **kwargs) -> str:
        """
        Retrieves and formats a prompt for a given task.

        Args:
            task_name (str): The name of the task (e.g., 'intent_classifier').
            **kwargs: The values to substitute into the prompt's placeholders.
                      NOTE: The required kwargs are dynamic and depend on the task.
                      To see the placeholders for a specific task, inspect the
                      'prompts.json' file.

        Returns:
            str: The formatted prompt string.

        Raises:
            ValueError: If the task is not found in the prompts file.
        """
        if task_name not in self.prompts:
            raise ValueError(f"Task '{task_name}' not found in prompts file.")

        template = self.prompts[task_name]['template']
        return template.format(**kwargs)
