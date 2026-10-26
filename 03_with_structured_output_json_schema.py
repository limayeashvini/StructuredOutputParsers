"""
03_with_structured_output_json_schema.py
----------------------------------------
Concept: Structured output using external JSON Schema
LangChain Feature: with_structured_output()

This script demonstrates:
- How to load a JSON Schema from a file.
- How to enforce model responses to match that schema.
- How to output structured, validated JSON results.

Example Use Case:
Extract key book information based on a defined schema stored externally.
"""

import json
from utils import get_llm


# ==========================================
# Setup
# ==========================================
base_llm = get_llm(temperature=0.7, max_tokens=400)


# ==========================================
# Step 1: Load JSON Schema from file
# ==========================================
schema_path = "book_schema.json"

with open(schema_path, "r") as f:
    raw_schema = json.load(f)

# Ensure OpenAI-compatible schema structure
schema = {
    "name": raw_schema.get("name", "BookInfo"),
    "description": raw_schema.get("title", "Book Information Schema"),
    # 👈 wrap actual schema under 'parameters' (minus file-level metadata keys
    # like '$schema'/'name', which Gemini rejects with "not supported" warnings)
    "parameters": {k: v for k, v in raw_schema.items() if k not in ("$schema", "name")},
}
'''
This reformats your schema to match what OpenAI expects.

* Why this wrapping is needed:
OpenAI's structured output feature requires a specific format:
- name = Name of the function/schema (top-level identifier)
- description = What this schema is for (helps the AI understand)
- parameters = The actual JSON schema rules (nested inside)

Think of it like mailing a package:
- name = "Fragile Electronics" (label on the box)
- description = "Contains a laptop and charger" (description on customs form)
- parameters = The actual items inside the box (the real content)
'''

# ==========================================
# Step 2: Wrap model with structured output enforcement
# ==========================================
llm = base_llm.with_structured_output(schema=schema)
''' This connects your formatted schema to the LLM. '''

# ==========================================
# Step 3: Define prompt and run
# ==========================================
prompt = (
    "Extract structured book information from this description:\n\n"
    "'In 1984, George Orwell depicts a dystopian future where government surveillance "
    "and censorship dominate society. The novel explores themes of freedom, control, "
    "and resistance against totalitarianism.'"
)

print("📘 Structured Output with External JSON Schema")
print("---------------------------------------------")
print("🔹 Loaded Schema:", schema_path)
print("\n🔹 Prompt:\n")
print(prompt)
print("\n🔹 Model Structured Response:\n")

response = llm.invoke(prompt)
print(json.dumps(response, indent=2))
