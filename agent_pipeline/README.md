# Agent Pipeline
In this part, we explore how the performance of target models detecting implicit bias could be improved by using an LLM-based agent system. 

Here, we use Gemini-1.5-Pro-latest as our core model in the agent system.
![alt text](image.png)

## Code Structure

```plaintext
agent_pipeline/
│
├── config/
│   └── settings.py
│
├── agents/
│   ├── prompt_refiner.py
│   ├── target_model.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_refinement_loop.py
│
├── utils/
│   ├── evaluation.py
│   ├── io.py
│
├── results/
│   ├── Llmama-3.3-70B-Instruct
│   ├── qwen2.5-7b-instruct
│   ├── ...
│
├── .env
├── requirements.txt
└── README.md

## How to run the code?
### Create your Python environment:
```bash
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

### Insert API keys and URL in .env:
NOVITA_API_KEY="<your_novita_api_key>"
NOVITA_API_URL="https://api.novita.ai/v3/openai"
GOOGLE_API_KEY="<your_gemini_api_key>"

### Get ready to start prompting:
# config/settings.py
1. Set the TARGET_MODEL_NAME with model ID
2. Set the REASON for different versions of prompts
# scripts/run_baseline.py
```bash
python scripts/run_baseline.py

### Start Refinement Loop:
# scripts/run_refinement_loop.py
1. Set the BASELINE_TIMESTAMP of the baseline run which we want to refine (is to be found in summary.json)
2. LOOP is set to 1 by default. Change this manually for further loops.
```bash
python scripts/run_refinement_loop.py