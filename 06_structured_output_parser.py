"""
06_structured_output_parser.py
------------------------------
Concept: Enforcing schema-based structured responses
LangChain Classes: ResponseSchema, StructuredOutputParser

This script demonstrates:
- How to define field-level schemas for LLM output.
- How to build a StructuredOutputParser to enforce consistent response formatting.
- How to show format instructions (for prompt injection into the model).
- How to parse and print structured JSON output.

Example Use Case:
Extract key details (product name, category, price, sentiment) from a customer review.
"""

import json

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from utils import get_llm


# ==========================================
# Setup
# ==========================================
llm = get_llm(temperature=0.3, max_tokens=400)


# ==========================================
# Step 1: Define Schema for structured output
# ==========================================
schemas = [
    ResponseSchema(name="product_name", description="Name of the product being reviewed."),
    ResponseSchema(name="category", description="Product category or type."),
    ResponseSchema(name="price", description="Approximate price mentioned in the review, if any."),
    ResponseSchema(name="sentiment", description="Overall sentiment of the review. One of: positive, negative, neutral."),
    ResponseSchema(name="summary", description="One-line summary of the review.")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

print("-----------------------------------------------------------")
print(format_instructions)
print("-----------------------------------------------------------")

# ==========================================
# Step 2: Define prompt
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["review"],
    template=(
        "You are an assistant that extracts structured information from product reviews.\n\n"
        "Customer Review:\n{review}\n\n"
        "Please return your answer in the following format:\n{format_instructions}"
    )
)

# Create final formatted prompt
customer_review = (
    "I recently bought the Sony WH-1000XM5 headphones, and they are absolutely amazing! "
    "The sound quality is superb, noise cancellation is unbeatable, though they’re quite expensive at around $400."
)

formatted_prompt = prompt_template.format(
    review=customer_review, format_instructions=format_instructions
)

# ==========================================
# Step 3: Run model
# ==========================================
print("🧾 Structured Output Parser Demonstration")
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
    print("🔹 Parsed Structured Output:\n")
    print(json.dumps(parsed_output, indent=2))
except Exception as e:
    print(f"[Parsing Error] {e}")