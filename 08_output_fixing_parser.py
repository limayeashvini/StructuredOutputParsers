"""
08_output_fixing_parser.py
--------------------------
Concept: Automatically repairing malformed outputs.
LangChain Class: OutputFixingParser

This script demonstrates:
- How to wrap a base parser (e.g., PydanticOutputParser) with an OutputFixingParser.
- How LangChain auto-fixes invalid JSON or schema mismatches.
- How to compare the raw vs fixed outputs.

Example Use Case:
A model returns slightly malformed JSON — OutputFixingParser repairs it automatically.
"""

import json
from pydantic import BaseModel, Field

from langchain_classic.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.prompts import PromptTemplate
from utils import get_llm


# ==========================================
# Setup
# ==========================================
base_llm = get_llm(temperature=0.5, max_tokens=400)


# ==========================================
# Step 1: Define Pydantic schema
# ==========================================
class MovieDetails(BaseModel):
    title: str = Field(..., description="Movie title.")
    year: int = Field(..., description="Release year.")
    genre: str = Field(..., description="Genre of the movie.")
    rating: float = Field(..., description="IMDB rating (0-10).")


# Base Pydantic parser
pydantic_parser = PydanticOutputParser(pydantic_object=MovieDetails)

# OutputFixingParser wraps the base parser
fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=base_llm)

# For LLM prompt hinting
format_instructions = pydantic_parser.get_format_instructions()


# ==========================================
# Step 2: Create intentionally “messy” output scenario
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["movie"],
    template=(
        "Extract structured details about the movie '{movie}'.\n\n"
        "IMPORTANT: You MUST intentionally create a JSON syntax error!\n"
        "For example:\n"
        "- Missing a comma between fields\n"
        "- Unclosed quote around a string\n"
        "- Extra trailing comma\n"
        "- Missing a required field entirely\n\n"
        "DO NOT output valid JSON. Make a deliberate mistake.\n\n"
        "Here's the schema you should have followed (but remember, MAKE AN ERROR):\n"
        "{format_instructions}"
    )
)

movie = "The Matrix (1999)"
formatted_prompt = prompt_template.format(movie=movie, format_instructions=format_instructions)


# ==========================================
# Step 3: Get raw model output and auto-fix it
# ==========================================
print("🧩 Output Fixing Parser Demonstration")
print("-------------------------------------")
print("🔹 Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n-------------------------------------\n")

# Get raw response (may be slightly invalid)
raw_response = base_llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n-------------------------------------\n")

# Use fixing parser — it re-prompts LLM internally to repair invalid JSON
try:
    fixed_output = fixing_parser.invoke(raw_response.content)
    print("🔹 Fixed & Parsed Output (Pydantic validated):\n")
    print(json.dumps(fixed_output.model_dump(), indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
