"""
11_yaml_or_custom_parser.py
----------------------------
Concept: Parsing non-JSON structured outputs (YAML / custom text)
LangChain Classes: BaseOutputParser (custom subclass example)

This script demonstrates:
- How to handle LLM responses formatted as YAML instead of JSON.
- How to build a simple custom parser that converts YAML → Python dict.
- How to register your own parsing logic (extremely useful for API configs, prompts, etc.).

Example Use Case:
Extract product configuration data returned as YAML.
"""

import json
import yaml
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from utils import get_llm


# ==========================================
# 1️⃣ Setup
# ==========================================
llm = get_llm(temperature=0.3, max_tokens=400)


# ==========================================
# 2️⃣ Define a custom YAML parser
# ==========================================
class YAMLParser(BaseOutputParser):
    """Custom output parser that loads YAML strings into Python dicts."""
    # BaseOutputParser = The parent class from LangChain that all output parsers inherit from
    # Purpose: Extend LangChain's parser system with our own YAML-handling logic

    def _clean_text(self, text: str) -> str:
        """Remove markdown-style YAML fences and trim text."""
        # Strip common code fence markers
        if text.strip().startswith("```"):
            text = text.strip().strip("`")
        text = text.replace("```yaml", "").replace("```yml", "").replace("```", "").strip()
        return text
    '''
    What it cleans:
    Raw LLM Output <<->> After Cleaning
    ```yaml\nname: John\nage: 30\n``` <<->> name: John\nage: 30
    ```yml\nname: John\nage: 30\n``` <<->> name: John\nage: 30
    ```\nname: John\nage: 30\n``` <<->> name: John\nage: 30
    name: John\nage: 30 <<->> name: John\nage: 30

    Why this is needed:
    LLMs often wrap YAML in markdown code blocks but yaml.safe_load() 
    can't parse those backticks - they're not valid YAML!
    '''

    def parse(self, text: str) -> dict:
        cleaned_text = self._clean_text(text)
        try:
            return yaml.safe_load(cleaned_text)
        except yaml.YAMLError as e:
            print("⚠️ Raw text before failure:\n", cleaned_text)
            raise ValueError(f"Failed to parse YAML output: {e}")
    '''
    This is the core method that converts YAML text to a Python dictionary.

    What happens:
    - Clean the text (remove markdown fences)
    - Attempt to parse with PyYAML's safe_load()
    - Success → Returns a Python dictionary
    - Failure → Prints debug info and raises error        
    '''

    def get_format_instructions(self) -> str:
        return "Please respond in valid YAML format, without using markdown code fences (no ```yaml or ```)."


# Instantiate our parser
yaml_parser = YAMLParser()
format_instructions = yaml_parser.get_format_instructions()


# ==========================================
# 3️⃣ Define prompt expecting YAML output
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["product"],
    template=(
        "You are a structured data assistant.\n"
        "Extract detailed configuration for the following product in YAML format.\n\n"
        "Product: {product}\n\n"
        "{format_instructions}"
    ),
)

product_name = "MacBook Air M3 (2024)"

formatted_prompt = prompt_template.format(
    product=product_name, format_instructions=format_instructions
)


# ==========================================
# 4️⃣ Run LLM and parse YAML output
# ==========================================
print("🧾 YAML / Custom Output Parser Demonstration")
print("---------------------------------------------")
print(f"🔹 Product: {product_name}\n")
print("🔹 Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n---------------------------------------------\n")

raw_response = llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n---------------------------------------------\n")

try:
    parsed_output = yaml_parser.invoke(raw_response.content)
    print("🔹 Parsed YAML → Python dict:\n")
    print(json.dumps(parsed_output, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
