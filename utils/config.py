# utils/config.py
import os

# Streamlit Cloud puts these into the environment for you
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY  = os.getenv("PERPLEXITY_API_KEY")
