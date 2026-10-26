# Structured Output & Output Parsers with LangChain

Hands-on examples showing how to get **structured, reliable data** out of LLMs using LangChain — from native `with_structured_output()` to classic output parsers, auto-fixing, few-shot prompting, and custom (YAML) parsers.

Each script is standalone, numbered in learning order, and works with either **Google Gemini** or **OpenAI**, switchable from a single config file.

---

## 📂 Project Structure

| File | Concept | Key LangChain Feature |
|------|---------|-----------------------|
| `01_with_structured_output_typed_dict.py` | Structured output from a `TypedDict` schema | `with_structured_output()` |
| `02_with_structured_output_pydantic.py` | Structured output validated by a Pydantic model | `with_structured_output()` |
| `03_with_structured_output_json_schema.py` | Structured output from an external JSON Schema file | `with_structured_output()` |
| `04_str_output_parser.py` | Plain-text extraction | `StrOutputParser` |
| `05_json_output_parser.py` | Parsing JSON text into a Python dict | `JsonOutputParser` |
| `06_structured_output_parser.py` | Field-level schemas + format instructions | `ResponseSchema`, `StructuredOutputParser` |
| `07_pydantic_output_parser.py` | Typed, validated parsing with Pydantic | `PydanticOutputParser` |
| `08_output_fixing_parser.py` | Automatically repairing malformed output | `OutputFixingParser` |
| `09_combined_chain_structured_parser.py` | End-to-end pipeline: prompt → model → parser | LCEL `prompt \| llm \| parser` |
| `10_few_shot_structured_parser.py` | Few-shot examples for consistent formatting | `StructuredOutputParser` |
| `11_yaml_or_custom_parser.py` | Parsing YAML via a custom parser class | `BaseOutputParser` subclass |
| `pydantic_features_demo.py` | Pydantic v2 basics (validation, `Field`, `Annotated`) — no LLM needed | — |

**Supporting files**

- `utils.py` — shared `get_llm()` helper that builds the chat model from `config.json` (used by every numbered script).
- `config.json` — selects the provider and model settings.
- `book_schema.json` — JSON Schema used by script `03`.
- `requirements.txt` — pinned dependencies.
- `.env` — your API keys (not committed).

---

## ⚙️ Setup

**1. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API keys** — create a `.env` file in the project folder:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

You only need the key for the provider you use.

**4. Choose a provider** in `config.json`:

```json
{
  "provider": "gemini",
  "openai": { "model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 2000 },
  "gemini": { "model": "models/gemini-3.5-flash-lite", "temperature": 0.7, "max_output_tokens": 200 }
}
```

Set `"provider"` to `"gemini"` or `"openai"`. Model, temperature, and token limits are read from the matching section.

---

## ▶️ Running the Examples

Run any script directly:

```bash
python 01_with_structured_output_typed_dict.py
python 06_structured_output_parser.py
python 11_yaml_or_custom_parser.py
```

Each script prints the prompt, the raw model response, and the parsed structured result.

---

## 🧩 How the LLM Is Created

All scripts get their model from `utils.py`:

```python
from utils import get_llm

llm = get_llm(temperature=0.3, max_tokens=400)
```

`get_llm()` loads `.env`, reads `config.json`, and returns a `ChatGoogleGenerativeAI` or `ChatOpenAI` instance. The `temperature` / `max_tokens` arguments are **fallbacks** used only when `config.json` doesn't define them.

---

## 🧠 Two Approaches to Structured Output

| | Native (`with_structured_output`) — scripts 01–03 | Output Parsers — scripts 04–11 |
|---|---|---|
| How it works | Model's built-in function/JSON mode enforces the schema | Format instructions are added to the prompt; the text reply is parsed afterwards |
| Reliability | High — provider enforces the structure | Depends on the model following instructions (see `OutputFixingParser`) |
| Model support | Requires a model with tool/structured-output support | Works with any text model |
| Flexibility | JSON only | JSON, YAML, plain text, or any custom format |

---

## 📝 Notes

- **LangChain 1.x:** classic parsers (`ResponseSchema`, `StructuredOutputParser`, `PydanticOutputParser`, `OutputFixingParser`) now live in the `langchain-classic` package:
  ```python
  from langchain_classic.output_parsers import StructuredOutputParser
  ```
  Core parsers (`StrOutputParser`, `JsonOutputParser`, `BaseOutputParser`) come from `langchain_core.output_parsers`.
- **Gemini model retirement:** if you see `404 ... model is no longer available`, update the `model` value in `config.json` to the version named in the error.
- **Truncated output:** if JSON parsing fails because the response is cut off, raise `max_output_tokens` / `max_tokens` in `config.json`.
- **Gemini & JSON Schema:** Gemini rejects metadata keys such as `$schema` and `name` inside a schema; script `03` strips them before sending.
