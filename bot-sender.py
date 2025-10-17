import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Read session token from temp folder
base_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(base_dir, "temp")
session_file = os.path.join(temp_dir, "session_token.txt")

if not os.path.exists(session_file):
    raise FileNotFoundError("Session token file not found. Make sure the bot is logged in.")

with open(session_file, "r") as f:
    session_token = f.read().strip()

# Host and room info
SPEAKEASY_HOST = os.getenv("SPEAKEASY_HOST")  # same host as your Agent uses
ROOM_ID = input("Enter room ID: ")
MESSAGE = input("Enter message to send: ")

# Send the message
url = f"{SPEAKEASY_HOST}/api/room/{ROOM_ID}"
res = requests.post(url, params={"session": session_token}, data=MESSAGE.encode("utf-8"))

if res.ok:
    print("Message sent successfully!")
else:
    print(f"Failed to send message: {res.status_code} - {res.text}")
