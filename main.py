import time
import os
import shutil
import atexit

from app.core import App
from speakeasypy import Chatroom, EventType, Speakeasy
from dotenv import load_dotenv
from config import AGENT_CONFIG
# Load variables from .env file
load_dotenv()

class Agent:
    def __init__(self, username, password, mode=None, preload_strategy=None):
        print(f"\n{100*'-'}\nInitializing Agent...\n{100*'-'}")
        print(f"Config: {AGENT_CONFIG}\n{100*'-'}")

        # Use config mode if no mode is provided
        self.mode = mode if mode is not None else AGENT_CONFIG["mode"]
        
        # Use dataset path from config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_dir, AGENT_CONFIG["dataset_path"])
        embeddings_path = os.path.join(base_dir, AGENT_CONFIG["embeddings_path"])
        
        # Determine preload strategy
        preload_strategy = preload_strategy if preload_strategy is not None else AGENT_CONFIG["preload_strategy"]
        
        # Initialize App with preloading configuration
        self.app = App(dataset_path, embeddings_path, preload_strategy=preload_strategy, mode=self.mode)
        
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=AGENT_CONFIG["speakeasy_host"], username=username, password=password)
        self.session_token = self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

        # make temporary folder
        self.temp_dir = os.path.join(base_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.session_file = os.path.join(self.temp_dir, "session_token.txt")
        with open(self.session_file, "w") as f:
            f.write(self.session_token)

        atexit.register(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

        # Prepare logging
        self.logs_dir = os.path.join(base_dir, "evidence")
        os.makedirs(self.logs_dir, exist_ok=True)

        # Log filename: YYYY-MM-DD-sessiontoken.log
        self.log_file = os.path.join(
            self.logs_dir,
            f"{time.strftime('%Y-%m-%d')}-{self.session_token}.log"
        )
        self.speakeasy.register_callback(self.on_new_message, EventType.MESSAGE)
        self.speakeasy.register_callback(self.on_new_reaction, EventType.REACTION)

    def listen(self):
        """Start listening for events."""
        self.speakeasy.start_listening()

    def log_event(self, chatroom_id: str, question: str, answer: str, response_time: float):
        """Write structured logs to file, including response time."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = (
            f"[{timestamp}] - {self.session_token} - {chatroom_id} - "
            f"question: \"{question}\" - answer: \"{answer}\" - "
            f"response_time: {response_time:.2f} s\n"
        )
        try:
            with open(self.log_file, "a", encoding="utf-8") as log:
                log.write(log_entry)
        except Exception as e:
            print(f"Logging error: {e}")


    def on_new_message(self, message : str, room : Chatroom):
        """Callback function to handle new messages."""
        try:
            print(f"{100*'-'}\nNew message in room {room.room_id}: {message}\n{100*'-'}")
            start = time.time()
            answer = self.app.get_answer(message=message, mode=self.mode)
            response_time = time.time() - start
            print(f"{100*'-'}\nAnswer from Bot (in {response_time:.2f} s): '{answer}'.\n{100*'-'}")
            room.post_messages(f"{answer}")
            self.log_event(room.room_id, message, answer, response_time)
        except Exception as e:
            print(f"{100*'-'}\nError: {e}\n{100*'-'}")
            room.post_messages(f"I cannot answer your question. Please try again!")
            return

    def on_new_reaction(self, reaction : str, message_ordinal : int, room : Chatroom): 
        """Callback function to handle new reactions."""
        try:
            print(f"\n{100*'-'}\nNew reaction '{reaction}' on message #{message_ordinal} in room {room.room_id}")
            print(f"\nMessage data: {self.get_message_by_ordinal(room, message_ordinal)}\n{100*'-'}")  # Get the specific message that received the reaction as JSON
            
            # Send a reaction based on reaction type ChatMessageReactionType.STAR, ChatMessageReactionType.THUMBS_DOWN, ChatMessageReactionType.THUMBS_UP
            # TODO: Implement agent logic here, for now, we just send an echo of the reaction.
            if str(reaction) == 'ChatMessageReactionType.STAR':
                room.post_messages(f"Thanks for your reaction: Star")
            elif str(reaction) == 'ChatMessageReactionType.THUMBS_DOWN':
                room.post_messages(f"Thanks for your reaction: Thumbs Down")
            elif str(reaction) == 'ChatMessageReactionType.THUMBS_UP':
                room.post_messages(f"Thanks for your reaction: Thumbs Up")
        except ValueError as ve:
            print(f"{100*'-'}\nValueError: {ve}\n{100*'-'}")
            room.post_messages(f"I cannot find the message you reacted to. Please try again!")
            return
        except Exception as e:
            print(f"{100*'-'}\nError: {e}\n{100*'-'}")
            room.post_messages(f"I cannot process your reaction. Please try again!")
            return

    def get_message_by_ordinal(self, room: Chatroom, ordinal: int):
        """
        Find and return a specific message by its ordinal number as JSON.
        
        Args:
            room (Chatroom): The chatroom to search in
            ordinal (int): The ordinal number of the message to find
            
        Returns:
            dict: JSON object with message attributes
            
        Raises:
            ValueError: If message with the specified ordinal is not found
        """
        try:
            all_messages = room.get_messages(only_partner=False, only_new=False)
            for message in all_messages:
                if message.ordinal == ordinal:
                    return {
                        "ordinal": message.ordinal,
                        "content": message.message,
                        "author_alias": message.author_alias,
                        "timestamp": message.time_stamp,
                        "recipients": message.recipients,
                        "read": message.read
                    }
            raise ValueError(f"No message found with ordinal {ordinal}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Error finding message by ordinal {ordinal}: {e}")

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    username = os.getenv("SPEAKEASY_USERNAME")
    password = os.getenv("SPEAKEASY_PASSWORD")
    # Mode is now read from config.py, but can be overridden if needed
    conversational_bot = Agent(username, password)  # Uses mode from config / conversational_bot = Agent(username, password, mode=2) -> overrides config mode
    conversational_bot.listen()