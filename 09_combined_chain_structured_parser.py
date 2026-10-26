"""
09_combined_chain_structured_parser.py
--------------------------------------
Concept: Building an end-to-end structured output chain.
LangChain Components: PromptTemplate | ChatModel | StructuredOutputParser

This script demonstrates:
- How to combine Prompt, Model, and Parser into one pipeline using LangChain’s chain syntax.
- How to pass dynamic input (movie review) and get structured data.
- A clean, reusable pattern for structured data extraction pipelines.

Example Use Case:
Extract key information (title, director, rating, sentiment) from movie reviews.
"""

import json
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from utils import get_llm


# ==========================================
# Setup
# ==========================================
llm = get_llm(temperature=0.5, max_tokens=400)


# ==========================================
# Step 1: Define schema fields
# ==========================================
schemas = [
    ResponseSchema(name="title", description="Title of the movie."),
    ResponseSchema(name="director", description="Director of the movie."),
    ResponseSchema(name="release_year", description="Year when the movie was released."),
    ResponseSchema(name="rating", description="IMDB-style numeric rating out of 10."),
    ResponseSchema(name="sentiment", description="Sentiment of the review (positive/negative/neutral).")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()


# ==========================================
# Step 2: Build the chain (Prompt → Model → Parser)
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "You are an assistant that extracts structured movie information from text.\n\n"
        "Customer Review:\n{review}\n\n"
        "Please return your answer in the following format:\n{format_instructions}"
    )
)

# Combine all three into a single chain
chain = prompt_template | llm | parser


# ==========================================
# Step 3: Test with input
# ==========================================
review_text = (
    "I just watched Oppenheimer and it was absolutely stunning! "
    "Christopher Nolan outdid himself again — powerful storytelling, "
    "immersive sound design, and amazing performances. "
    "Easily a 9.5 out of 10 for me!"
)

print("🎬 Combined Structured Chain Demonstration")
print("------------------------------------------")
print("🔹 Review Text:\n")
print(review_text)
print("\n------------------------------------------\n")

# Run chain
try:
    result = chain.invoke({"review": review_text, "format_instructions": format_instructions})
    print("🔹 Parsed Structured Output:\n")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")