"""
07_pydantic_output_parser.py
----------------------------
Concept: Using Pydantic models for structured parsing
LangChain Class: PydanticOutputParser

This script demonstrates:
- How to define a Pydantic model to enforce field types and validation.
- How to use LangChain’s PydanticOutputParser to parse model outputs into that schema.
- How to show format instructions and parsed structured output.

Example Use Case:
Extract movie metadata (title, director, year, genre, rating) from a review.
"""

import json
from pydantic import BaseModel, Field

from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from utils import get_llm


# ==========================================
# Setup
# ==========================================
llm = get_llm(temperature=0.3, max_tokens=400)


# ==========================================
# Step 1: Define Pydantic schema
# ==========================================
class MovieInfo(BaseModel):
    """Schema for structured movie information."""
    title: str = Field(..., description="The movie's title.")
    director: str = Field(..., description="Name of the movie's director.")
    year: int = Field(..., description="Release year of the movie.")
    genre: str = Field(..., description="Movie genre or category.")
    rating: float = Field(..., description="IMDB-style numeric rating out of 10.")

# Initialize Pydantic output parser
parser = PydanticOutputParser(pydantic_object=MovieInfo)
format_instructions = parser.get_format_instructions()


# ==========================================
# Step 2: Create prompt
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "Extract the structured movie information from this review:\n\n"
        "{review}\n\n"
        "Return data in the exact format below:\n{format_instructions}"
    )
)

review_text = (
    "Christopher Nolan's Interstellar is a breathtaking sci-fi epic released in 2014. "
    "It stars Matthew McConaughey and explores love, survival, and time travel. "
    "A true masterpiece rated around 8.6 on IMDb."
)

formatted_prompt = prompt_template.format(
    review=review_text, format_instructions=format_instructions
)


# ==========================================
# Step 3: Generate and parse response
# ==========================================
print("🎬 Pydantic Output Parser Demonstration")
print("----------------------------------------")
print("🔹 Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n----------------------------------------\n")

raw_response = llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n----------------------------------------\n")


# ==========================================
# Step 4: Parse structured output
# ==========================================
try:
    parsed_output = parser.invoke(raw_response.content)
    print("🔹 Parsed Structured Output (validated Pydantic model):\n")
    # Compatible with Pydantic v2+
    print(json.dumps(parsed_output.model_dump(), indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")

