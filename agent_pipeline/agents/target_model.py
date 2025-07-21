import time
from openai import OpenAI, RateLimitError, APIError
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import (
    NOVITA_API_KEY, NOVITA_API_URL, OPENAI_API_KEY, GOOGLE_API_KEY,
    TARGET_MODEL_NAME, TARGET_MODEL_SOURCE
)

# Inference settings
stream = False
max_tokens = 2000
system_content = "Be a helpful assistant"
temperature = 1
top_p = 1
min_p = 0
top_k = 50
presence_penalty = 0
frequency_penalty = 0
repetition_penalty = 1

def call_target_model(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    attempt = 0

    while attempt < retries:
        try:
            if TARGET_MODEL_SOURCE == "openai":
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content

            elif TARGET_MODEL_SOURCE == "gemini":
                llm = ChatGoogleGenerativeAI(
                    model=TARGET_MODEL_NAME,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.0
                )
                response = llm.invoke(prompt)
                return response.content if hasattr(response, "content") else response

            elif TARGET_MODEL_SOURCE == "novita":
                client = OpenAI(
                    base_url=NOVITA_API_URL,
                    api_key=NOVITA_API_KEY,
                )
                response = client.chat.completions.create(
                    model=TARGET_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    stream=stream,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    extra_body={
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "min_p": min_p
                    }
                )
                return response.choices[0].message.content

            else:
                raise ValueError(f"❌ Unknown TARGET_MODEL_SOURCE: {TARGET_MODEL_SOURCE}. "
                                 f"Must be 'openai', 'novita', or 'gemini'.")

        except RateLimitError:
            wait = delay * (2 ** attempt)
            print(f"⚠️ Rate limit hit. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            attempt += 1

        except APIError as e:
            print(f"❌ APIError on attempt {attempt + 1}: {str(e)}")
            time.sleep(delay)
            attempt += 1

    raise RuntimeError("❌ Failed to get response after multiple retries due to rate limits or errors.")
