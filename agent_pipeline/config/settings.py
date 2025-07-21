import os
from dotenv import load_dotenv

load_dotenv()

# === API URL & KEYS ===
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
NOVITA_API_URL = os.getenv("NOVITA_API_URL")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Model SOURCES ===
CORE_AGENT_SOURCE = "" # or "openai"
TARGET_MODEL_SOURCE = "novita" # or "novita" or "openai" 

# === Model NAMES ===
PROMPT_REFINER_MODEL = "" # or "gpt-3.5-turbo"
TARGET_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct" # Edit the target model here manually

# === Experiment ===
REASON = False # Edit REASON manually to test different prompts
