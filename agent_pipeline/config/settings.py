import os
from dotenv import load_dotenv

load_dotenv()

# === API KEYS ===
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
NOVITA_API_URL = os.getenv("NOVITA_API_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Models ===
PROMPT_REFINER_MODEL = "gemini-1.5-pro-latest"
TARGET_MODEL_NAME = "qwen/qwen2.5-7b-instruct" # Edit the target model here manually

# === Experiment ===
REASON = True # Edit REASON manually to test different prompts
