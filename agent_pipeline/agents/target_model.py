from openai import OpenAI

from config.settings import NOVITA_API_KEY, NOVITA_API_URL, OPENAI_API_KEY, TARGET_MODEL_NAME, TARGET_OPENAI

stream = False # or False
max_tokens = 2000
system_content = "Be a helpful assistant"
temperature = 1
top_p = 1
min_p = 0
top_k = 50
presence_penalty = 0
frequency_penalty = 0
repetition_penalty = 1

def call_target_model(prompt: str) -> str:
    
    if TARGET_OPENAI: # Target model is from GPT series
        client = OpenAI(
        api_key=OPENAI_API_KEY
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "user", "content": prompt}
            ], 
            temperature=0.0
        )
        final_response = response.choices[0].message.content
        
    else: # Target model can be called using Novita API
        client = OpenAI(
            base_url=NOVITA_API_URL,
            api_key=NOVITA_API_KEY,
        )
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
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
        final_response = response.choices[0].message.content
    return final_response