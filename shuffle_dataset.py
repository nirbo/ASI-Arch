#!/usr/bin/env python3
"""
Shuffle JSONL dataset rows for better training diversity.
"""

import json
import random
import argparse
from pathlib import Path

def shuffle_jsonl_dataset(input_file: str, output_file: str = None, seed: int = 42):
    """
    Shuffle rows in a JSONL dataset file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (defaults to input_file with _shuffled suffix)
        seed: Random seed for reproducible shuffling
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_shuffled{input_path.suffix}")
    
    print(f"📂 Reading dataset from: {input_file}")
    
    # Load all rows
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    rows.append(row)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue
    
    print(f"📊 Loaded {len(rows)} samples")
    
    # Shuffle the rows
    print(f"🔀 Shuffling with seed {seed}...")
    random.shuffle(rows)
    
    # Write shuffled dataset
    print(f"💾 Writing shuffled dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully shuffled {len(rows)} samples")
    print(f"📁 Original: {input_file}")
    print(f"📁 Shuffled: {output_file}")
    
    # Show first few samples to verify shuffling
    print(f"\n🔍 First 3 samples after shuffling:")
    for i, row in enumerate(rows[:3]):
        if 'messages' in row and len(row['messages']) > 0:
            first_content = row['messages'][0].get('content', '')[:100]
            print(f"  {i+1}. {first_content}...")
        elif 'text' in row:
            print(f"  {i+1}. {row['text'][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Shuffle JSONL dataset rows")
    parser.add_argument("input_file", help="Input JSONL dataset file")
    parser.add_argument("--output", "-o", help="Output file (default: input_shuffled.jsonl)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"❌ Error: Input file '{args.input_file}' not found")
        return 1
    
    try:
        shuffle_jsonl_dataset(args.input_file, args.output, args.seed)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())