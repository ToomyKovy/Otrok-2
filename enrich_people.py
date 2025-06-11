#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.analyze import compare_with_manual
from utils.excel import save_to_xlsx
from utils.llm import LLM, extract_json_data

# CHOSEN_MODEL = 'http://192.168.2.157:8081/v1'  # Local Deepseek
DEFAULT_MODEL = 'gpt-4.1-2025-04-14'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Segment people from csv data.")
    parser.add_argument("csv_file", type=Path, help="CSV file to be processed")
    parser.add_argument("-l", "--limit", type=int, default=0, help="Process first N rows.")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="LLM model.")
    args = parser.parse_args()

    people_csv_file_exported_from_clay = Path(args.csv_file)

    with open('system_prompt.txt', 'r', encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()

    llm_wrapper = LLM(model=args.model, system_prompt=SYSTEM_PROMPT, log_level='DEBUG')


    people_df = pd.read_csv(people_csv_file_exported_from_clay)
    if args.limit > 0:
        people_df = people_df.head(args.limit)

    lp = len(people_df)
    people_df = people_df.dropna(subset="Name")
    if lp != len(people_df):
        print(f"Warning: {lp - len(people_df)} profiles were dropped due to missing 'Name' field.")
    print(f'This script will evaluate {len(people_df)} people profiles.')

    collected_completions = []
    for person_info in tqdm(people_df.to_dict('records')):
        data_to_call = person_info.copy()  # remove the 'Original segment' key
        original_segment = re.sub(r'[^\w ]', '', data_to_call.pop('Org Segment', '')).strip()

        out = llm_wrapper.call_api(str(data_to_call), max_tokens=8000)
        extracted_data: list = extract_json_data(out['completion'])

        collected_completions.append({
                "id": person_info["New Column"],
                "name": person_info['Name'],
                "original_segment": original_segment,
            } | {item["persona_type"]: item["score"] for item in extracted_data}
              | {item["persona_type"] + " - hint": item["short_reasoning"] for item in extracted_data}
        )

    collected_df = pd.DataFrame(collected_completions)

    print("Comparison with manual data:")
    for k, v in compare_with_manual(collected_df).items():
        prefix = f"➤ {k}:"
        print(f"{prefix:<18} {v:.2%}")

    # plot_persona_heatmap(collected_df, save_path='output/big_sample.svg', show=False)

    save_to_xlsx(collected_df, f"output/{people_csv_file_exported_from_clay.stem}.xlsx")

    # sub_df = collected_df[collected_df.name.isin([
    #     'Caesar Qulajen',
    #     'Nils Feigenwinter',
    #     'Nicole Worthington',
    #     'Branislav Glusac',
    #     'Nikolina Kolarić',
    #     'Vedran Mikić',
    #     'Eve Dunkley',
    #     'Salah Yahya',
    #     'Elliot Gay',
    #     'Rogier van Lammeren',
    # ])]
    # plot_persona_heatmap(sub_df, save_path='output/small_sample.svg', h=.5, show=False)
    #
    # sub_df = collected_df[collected_df.name.isin([
    #     'Martin Korbelar',
    #     'Lukas Hora',
    #     'Veronika Kincova',
    # ])]
    # plot_persona_heatmap(sub_df, save_path='output/sales_team.svg', h=.4, show=False)
