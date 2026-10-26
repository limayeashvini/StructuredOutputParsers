"""
utils.py
--------
Shared helper to build the chat model selected in config.json.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

CONFIG_PATH = Path(__file__).parent / "config.json"


def get_llm(temperature=0.7, max_tokens=300):
    """Return the LLM for the provider in config.json.

    `temperature` and `max_tokens` are fallbacks used only when config.json
    doesn't specify them.
    """
    load_dotenv()

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    provider = config["provider"]
    model_cfg = config[provider]

    if provider == "openai":
        return ChatOpenAI(
            model=model_cfg.get("model"),
            temperature=model_cfg.get("temperature", temperature),
            max_tokens=model_cfg.get("max_tokens", max_tokens),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_cfg.get("model"),
            temperature=model_cfg.get("temperature", temperature),
            max_output_tokens=model_cfg.get("max_output_tokens", max_tokens),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")
