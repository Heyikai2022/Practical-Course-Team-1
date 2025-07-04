import os
from dotenv import load_dotenv

load_dotenv()

# === API URL & KEYS ===
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
NOVITA_API_URL = os.getenv("NOVITA_API_URL")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Models ===
CORE_AGENT = "gemini" # or "openai"

PROMPT_REFINER_MODEL = "gemini-1.5-pro-latest" # or "gpt-3.5-turbo"
TARGET_MODEL_NAME = "" # Edit the target model here manually

# === Experiment ===
REASON = True # Edit REASON manually to test different prompts
TARGET_OPENAI = False # Set TARGET_OPENAI manually to True, if the tested target model is from GPT series
