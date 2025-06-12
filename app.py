# ── Streamlit front‑end for OTROK ─────────────────────────────────────────
"""
A one‑file Streamlit app that lets colleagues run the OTROK persona‑scoring
pipeline in the browser:

1. Upload a CSV of people (id, name, original_segment …). Works with either
   comma‑ or semicolon‑separated files (auto‑detect).
2. Optionally upload a custom system_prompt.txt (or fall back to a simple default).
3. Choose the LLM model + temperature and paste API keys in the sidebar (or let
   them auto‑fill from Streamlit **secrets**).
4. Click **Run OTROK** → every row is sent to the LLM, scores are parsed,
   heat‑map Excel + PNG are generated, and both summary ratios and downloads
   appear immediately.

Dependencies (add to requirements.txt):
    streamlit pandas openpyxl matplotlib seaborn openai tqdm python-dotenv

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import excel                               # persona palette & save_to_xlsx
from analyze import compare_with_manual     # overlap metrics
from excel import save_to_xlsx              # Excel heat‑map writer
from llm import LLM, extract_json_data      # LLM wrapper & JSON helper
from plotting import plot_persona_heatmap   # Matplotlib heat‑map

# ── Small helpers ────────────────────────────────────────────────────────

def _set_env(openai_key: str | None, pplx_key: str | None) -> None:
    """Populate ENV so llm.py picks the keys up via utils.config."""
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if pplx_key:
        os.environ["PERPLEXITY_API_KEY"] = pplx_key


@st.cache_data(show_spinner=False, ttl="1h")
def _persona_cols() -> list[str]:
    """Flat list of persona columns ordered by group."""
    return [c for cols in excel.col_types.values() for c in sorted(cols)]


def _read_csv_any_delim(uploaded_file) -> pd.DataFrame:
    """Read a CSV whose delimiter might be ',', ';', '\t', or '|'."""
    raw = uploaded_file.getvalue()
    # sniff first 4 KB to guess delimiter
    sample = raw[:4096].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except csv.Error:
        sep = ","  # fallback
    # rewind BytesIO for pandas
    buf = io.BytesIO(raw)
    df = pd.read_csv(buf, sep=sep)
    # normalise column names (strip whitespace / line breaks)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    return df


def _enrich_dataframe(df: pd.DataFrame, system_prompt: str, model: str, temp: float) -> pd.DataFrame:
    """Call the OTROK LLM pipeline row‑by‑row and return the scored DataFrame."""
    llm = LLM(model=model, system_prompt=system_prompt, log_level=3)
    persona_cols = _persona_cols()

    results: list[dict] = []
    progress = st.progress(0.0, text="Calling LLM …")

    for i, row in enumerate(df.to_dict("records"), start=1):
        # 👉 Customise this prompt if you need richer context per row.
        input_prompt = json.dumps(row, ensure_ascii=False)
        completion = llm.call_api(input_prompt=input_prompt, temperature=temp)
        data = extract_json_data(completion["completion"]) or {}
	if not data:                                                # <─ add
            st.error(                                               # <─ these
                f"Row {i}: LLM returned no parsable JSON:\n\n"
                f"{completion['completion']}"
            )    
        # guarantee all persona keys
        for p in persona_cols:
            data.setdefault(p, None)

        data.update({
            "id": row.get("id") or row.get("Id") or row.get("ID"),
            "name": row.get("name") or row.get("Name"),
            "original_segment": row.get("original_segment") or row.get("Original segment") or row.get("Original Segment"),
        })
        results.append(data)
        progress.progress(i / len(df), text=f"{i}/{len(df)} rows done")

    progress.empty()
    return pd.DataFrame(results)


# ── Streamlit UI ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="OTROK batch scorer", layout="wide")
    st.title("OTROK – Persona enrichment")

    # ↑—— Sidebar controls ————————————————————————————————
    with st.sidebar:
        st.header("Settings")
        # 🔑 Try to pre‑fill the fields from Streamlit secrets
        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=st.secrets.get("openai_api_key", ""),
            help="Leave blank to paste a one‑off key or rely on the server secret."
        )
        pplx_key = st.text_input(
            "Perplexity API key",
            type="password",
            value=st.secrets.get("perplexity_api_key", ""),
        )

        model_name = st.selectbox(
            "Model",
            [
                "gpt-4o-mini-2024-07-18",
                "gpt-4o-2024-11-20",
                "gpt-4.1-2025-04-14",
                "sonar",
                "sonar-reasoning",
            ],
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

        sys_prompt_file = st.file_uploader("Custom system prompt (.txt)", type=["txt"])
       system_prompt = (
         sys_prompt_file.read().decode("utf-8")
         if sys_prompt_file
         else Path("system_prompt.txt").read_text(encoding="utf-8")
         )
    _set_env(openai_key, pplx_key)

    # ↑—— Main panel ————————————————————————————————————————
    uploaded = st.file_uploader("Upload CSV of people", type=["csv"])
    if not uploaded:
        st.info("⬆️ Upload a CSV to get started.")
        return

    df_input = _read_csv_any_delim(uploaded)
    st.subheader("Preview of uploaded data (first 5 rows)")
    st.dataframe(df_input.head())

    if st.button("Run OTROK"):
        with st.spinner("Scoring personas … this may take a while!"):
            df_scored = _enrich_dataframe(df_input, system_prompt, model_name, temperature)

        # 1‣ Ratios -----------------------------------------------------------------
        ratios = compare_with_manual(df_scored)
        st.subheader("Overlap with original segment")
        st.write(ratios)

        # 2‣ Output artefacts --------------------------------------------------------
        tmp_dir = Path(tempfile.mkdtemp())
        xlsx_path = tmp_dir / "persona_scores.xlsx"
        png_path = tmp_dir / "heatmap.png"

        save_to_xlsx(df_scored, xlsx_path)
        plot_persona_heatmap(df_scored, png_path)

        # 3‣ Downloads --------------------------------------------------------------
        with open(xlsx_path, "rb") as f:
            st.download_button(
                "Download Excel heat‑map",
                f.read(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with open(png_path, "rb") as f:
            png_bytes = f.read()
            st.image(png_bytes, caption="Heat‑map", use_column_width=True)
            st.download_button(
                "Download PNG heat‑map",
                png_bytes,
                file_name=png_path.name,
                mime="image/png",
            )


if __name__ == "__main__":
    main()
