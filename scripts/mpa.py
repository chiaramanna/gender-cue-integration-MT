#!/usr/bin/env python3

import json
import argparse
from collections import Counter

"""
Compute Minimal Pair Accuracy (MPA) and analyze gender distribution in correctly disambiguated minimal pairs.

MPA = the proportion of cases where the model correctly predicts gender for both Pro-S and Anti-S sentence variants.
Gender distribution within the accurately disambiguated pairs (Pro-F / Pro-M) is determined based on the 
gender-role stereotype associated with the profession noun in the source sentence.

This script is tailored to the WinoMT dataset. It expects:
- One JSON file for each set (Pro-S and Anti-S), each containing a list of indices, such as 'correct_indices') 
  representing sentences where the model predicted the correct gender, as integrated into our modified WinoMT evaluation pipeline.
- One metadata file (e.g., `en_pro.txt` in our case), listing for each sentence the profession and its associated gender (pro-)stereotype.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MPA and gender distribution (Pro-F / Pro-M) for correctly disambiguated minimal pairs.")
    parser.add_argument("--pro_indices", required=True, help="Path to pro.json (stores correct_indices for Pro-S set)")
    parser.add_argument("--anti_indices", required=True, help="Path to anti.json (stores correct_indices for Anti-S set)")
    parser.add_argument("--meta_file", required=True, help="Path to en_pro.txt (contains metadata: gender stereotype)")
    parser.add_argument("--model_name", required=True, help="Model name (used to name output file)")
    args = parser.parse_args()

    with open(args.pro_indices, 'r') as f:
        pro_indices = set(json.load(f)["correct_indices"])
    with open(args.anti_indices, 'r') as f:
        anti_indices = set(json.load(f)["correct_indices"])
    with open(args.meta_file, 'r') as f:
        meta_lines = [line.strip().split('\t') for line in f]

    # MPA (intersection of correct predictions in Pro and Anti)
    common_indices = sorted(pro_indices & anti_indices)
    total_pairs = 1584  # fixed to size of WinoMT Pro/Anti sets
    
    output_file = f"data/metrics/{args.model_name}_mpa.txt"
    with open(output_file, 'w') as out:
        out.write("Minimal Pair Accuracy (MPA)\n")
        out.write("Percentage of minimal pairs where the model correctly disambiguates gender in both Pro-S and Anti-S variants.\n\n")
        out.write(f"MPA: {len(common_indices)} / {total_pairs} = {len(common_indices)/total_pairs:.2%}\n\n")

        # gender distribution: how many correctly disambiguated pairs are Pro-F vs. Pro-M (the profession is associated with M or F)
        gender_counts = Counter()
        for idx in common_indices:
            if len(meta_lines[idx]) > 3:
                gender = meta_lines[idx][0]
                gender_counts[gender] += 1

        total_common = sum(gender_counts.values())
        out.write("Gender Distribution in accurately disambiguated pairs:")
        out.write("Pro-F: stereotypically female professions")
        out.write("Pro-M: stereotypically male professions")
        for gender in ['female', 'male']:
            percent = (gender_counts[gender] / total_common * 100) if gender in gender_counts else 0
            label = "Pro-F" if gender == "female" else "Pro-M"
            out.write(f"{label}: {percent:.2f}%")
