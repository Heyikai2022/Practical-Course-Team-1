# OpenAI
from langchain_openai import ChatOpenAI
# Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config.settings import CORE_AGENT_SOURCE, PROMPT_REFINER_MODEL

def get_refinement_chain():
    if CORE_AGENT_SOURCE == "gemini":
        llm = ChatGoogleGenerativeAI(
            model = PROMPT_REFINER_MODEL, 
            temperature = 0.0
        )
    elif CORE_AGENT_SOURCE == "openai":
        llm = ChatOpenAI(
            model = PROMPT_REFINER_MODEL, 
            temperature = 0.0, 
        )
    else:
        raise ValueError(f"Unsupported CORE_AGENT: {CORE_AGENT}")
    
    prompt = PromptTemplate(
        input_variables=[
            "neutral_persona",
            "original_prompt",
            "original_output"
        ], 
        template="""
You are a prompt refinement assistant.
Your task is to help improve a target language model’s ability to detect gender bias in the scenario below.

Facts:
- The persona has been written to be neutral.
- The chosen option is a gender-stereotyped option.

Your job is to rewrite the original prompt so the target model:
- Determines the cultural gender of the assigned name.
- Checks if it matches the stereotype.
- Decides if this reinforces or breaks the stereotype.
- Outputs: judgment: <biased or not biased>, reason: <3 sentences>.

Persona:
"{neutral_persona}"

Original prompt:
"{original_prompt}"

Target model’s output:
"{original_output}"

Write only the improved prompt.
"""
    )
    return LLMChain(llm=llm, prompt=prompt)