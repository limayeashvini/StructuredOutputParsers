"""
10_few_shot_structured_parser.py
---------------------------------
Concept: Teaching structured response formatting through examples.
LangChain Classes: StructuredOutputParser, ResponseSchema

This script demonstrates:
- How to include few-shot examples with schema-based structured output.
- How to guide the LLM toward consistent structured responses.
- Compare model reliability with few-shot prompting vs zero-shot.

Example Use Case:
Extract structured movie data after showing example patterns.
"""

import json
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from utils import get_llm


# ==========================================
# Setup
# ==========================================
llm = get_llm(temperature=0.4, max_tokens=400)


# ==========================================
# Step 1: Define schema
# ==========================================
schemas = [
    ResponseSchema(name="title", description="Movie title."),
    ResponseSchema(name="genre", description="Movie genre or category."),
    ResponseSchema(name="year", description="Year of release."),
    ResponseSchema(name="sentiment", description="Overall sentiment (positive/negative/neutral)."),
    ResponseSchema(name="summary", description="Short one-line summary of the review.")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()


# ==========================================
# Step 2: Few-shot examples
# ==========================================

few_shot_examples = """
Example 1:
Review: "I loved The Shawshank Redemption. It's inspiring and emotional, truly timeless."
Response:
{{
  "title": "The Shawshank Redemption",
  "genre": "Drama",
  "year": 1994,
  "sentiment": "positive",
  "summary": "A deeply emotional and uplifting story of hope and friendship."
}}

Example 2:
Review: "Avatar was visually stunning but too long and predictable."
Response:
{{
  "title": "Avatar",
  "genre": "Science Fiction",
  "year": 2009,
  "sentiment": "neutral",
  "summary": "A visually spectacular film with a weak storyline."
}}
"""


# ==========================================
# Step 3: Prompt with examples
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "You are an AI that extracts structured data from movie reviews.\n\n"
        "Here are some examples of the correct response format:\n"
        f"{few_shot_examples}\n"
        "Now extract data from this new review and respond in the same structured JSON format below:\n"
        "{review}\n\n"
        "Make sure your output follows this structure:\n{format_instructions}"
    )
)

# ==========================================
# Step 4: Run structured chain
# ==========================================
review_text = (
    "I finally watched Joker — dark, intense, and haunting. Joaquin Phoenix was phenomenal. "
    "It’s definitely one of the best character studies in recent years!"
)

print("🎥 Few-Shot Structured Output Demonstration")
print("-------------------------------------------")
print("🔹 Review Text:\n")
print(review_text)
print("\n-------------------------------------------\n")

# Generate formatted prompt
formatted_prompt = prompt_template.format(review=review_text, format_instructions=format_instructions)
print("🔹 Full Prompt Sent to Model:\n")
print(formatted_prompt)
print("\n-------------------------------------------\n")

# Invoke the model
raw_response = llm.invoke(formatted_prompt)

print("🔹 Raw Model Response:\n")
print(raw_response)
print("\n-------------------------------------------\n")

# Parse structured response
try:
    parsed_output = parser.invoke(raw_response.content)
    print("🔹 Parsed Structured Output:\n")
    print(json.dumps(parsed_output, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")
