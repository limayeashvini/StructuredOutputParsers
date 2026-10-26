# """
# 05_json_output_parser.py
# ------------------------
# Concept: Extracting and parsing JSON-like outputs
# LangChain Class: JSONOutputParser

# This script demonstrates:
# - How to guide an LLM to output data in JSON format.
# - How to use JSONOutputParser to convert text into a valid Python dict.
# - How to display both the raw and parsed results.

# Example Use Case:
# Extract movie metadata (title, genre, release_year, rating) from a natural-language prompt.
# """

import json

from langchain_core.prompts import PromptTemplate
from utils import get_llm
# ==========================================
# FIX: Change 'JSONOutputParser' to 'JsonOutputParser'
# Note: You can also use 'from langchain_core.output_parsers import JsonOutputParser'
# ==========================================
from langchain_core.output_parsers.json import JsonOutputParser 


# ==========================================
# Setup
# ==========================================
llm = get_llm(temperature=0.3, max_tokens=300)


# ==========================================
# Step 1: Define JSON-style prompt
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["movie_name"],
    template=(
        "Provide structured information about the movie '{movie_name}'. "
        "Respond ONLY in valid JSON with the following fields:\n"
        "title, genre, release_year, main_actor, imdb_rating.\n\n"
        "Example:\n"
        "{{\"title\": \"Inception\", \"genre\": \"Sci-Fi\", \"release_year\": 2010, \"main_actor\": \"Leonardo DiCaprio\", \"imdb_rating\": 8.8}}"
    )
)


parser = JsonOutputParser()


# ==========================================
# Step 3: Run the code and show outputs
# ==========================================
movie = "Interstellar"

print("🎬 JSON Output Parsing Demonstration")
print("------------------------------------")
print(f"🔹 Input Movie: {movie}\n")

# Step 3a: Get raw model output (before parsing)
formatted_prompt = prompt_template.format(movie_name=movie)
# Assuming LLM initialization was successful, this will now run
raw_response = llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n------------------------------------\n")

# Step 3b: Parsed JSON Output
try:
    # This now uses the corrected 'parser' object
    parsed_output = parser.invoke(raw_response) 
    print("🔹 Parsed JSON Output:\n")
    print(json.dumps(parsed_output, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")