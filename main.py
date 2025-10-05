import time
import os

from app.core import App
from speakeasypy import Chatroom, EventType, Speakeasy
from dotenv import load_dotenv
# Load variables from .env file
load_dotenv()

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'


class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, "dataset", "graph.nt")
        self.app = App(dataset_path, mode=1)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def on_new_message(self, message : str, room : Chatroom):
        """Callback function to handle new messages."""
        print(f"New message in room {room.room_id}: {message}")
        # Implement your agent logic here, e.g., respond to the message.
        room.post_messages(f"Received your message: '{message}'.")
        answer = self.app.post_message(message=message)
        room.post_messages(f"Answer from Bot: '{answer}'.")

    def on_new_reaction(self, reaction : str, message_ordinal : int, room : Chatroom): 
        """Callback function to handle new reactions."""
        print(f"New reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
        # Implement your agent logic here, e.g., respond to the reaction.
        room.post_messages(f"Thanks for your reaction: '{reaction}'")

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    username = os.getenv("SPEAKEASY_USERNAME")
    password = os.getenv("SPEAKEASY_PASSWORD")
    demo_bot = Agent(username, password)
    demo_bot.listen()