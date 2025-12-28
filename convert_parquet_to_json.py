#!/usr/bin/env python3
"""Convert parquet file to entry_dict.json format"""

import ast
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow"])
    import pandas as pd

def parse_string_dict(value):
    """Parse string representation of dict to actual dict"""
    if isinstance(value, str):
        try:
            # Try to parse Python dict string (with single quotes)
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If that fails, try JSON parsing
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
    return value

def convert_parquet_to_entry_dict(parquet_path: str, output_path: str):
    """Convert parquet file to entry_dict.json format"""
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to dict format expected by the code
    # The format should be: {sample_index: entry_dict, ...}
    entry_dict = {}
    
    # Process each row
    for idx, row in df.iterrows():
        sample_id = str(row['sample_index']) if 'sample_index' in row else str(idx)
        entry = row.to_dict()
        
        # Parse string fields that should be dicts
        # For db_bench
        if 'answer_info' in entry and isinstance(entry['answer_info'], str):
            entry['answer_info'] = parse_string_dict(entry['answer_info'])
        if 'table_info' in entry and isinstance(entry['table_info'], str):
            entry['table_info'] = parse_string_dict(entry['table_info'])
        if 'skill_list' in entry and isinstance(entry['skill_list'], str):
            entry['skill_list'] = parse_string_dict(entry['skill_list'])
        # For os_interaction
        if 'initialization_command_item' in entry and isinstance(entry['initialization_command_item'], str):
            entry['initialization_command_item'] = parse_string_dict(entry['initialization_command_item'])
        if 'evaluation_info' in entry and isinstance(entry['evaluation_info'], str):
            entry['evaluation_info'] = parse_string_dict(entry['evaluation_info'])
        
        entry_dict[sample_id] = entry
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(entry_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(entry_dict)} entries to {output_path}")

if __name__ == "__main__":
    parquet_file = "/home/wangziyi/PoA/data/knowledge_graph/train-00000-of-00001.parquet"
    output_file = "./data/v0303/knowledge_graph/processed/v0409_tcc_9_to_12_first500/entry_dict.json"
    
    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    convert_parquet_to_entry_dict(parquet_file, output_file)
