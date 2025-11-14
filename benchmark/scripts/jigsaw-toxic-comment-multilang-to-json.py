import csv
import json
from collections import defaultdict


def convert_csv_to_json_by_language(input_file, output_prefix="toxicity"):
    """
    Convert CSV with toxicity data to JSON format, creating separate files per language.

    Args:
        input_file: Path to input CSV file
        output_prefix: Prefix for output JSON files (default: "toxicity")

    Returns:
        Dictionary mapping language codes to output file paths
    """
    # Dictionary to store messages by language
    messages_by_lang = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            lang = row['lang']

            # Create JSON object
            json_obj = {
                "message": row['comment_text'],
                "score": float(row['toxic']),
                "toxic": bool(int(row['toxic']))
            }

            messages_by_lang[lang].append(json_obj)

    # Write separate JSON file for each language
    output_files = {}
    for lang, messages in messages_by_lang.items():
        output_file = f"{output_prefix}_{lang}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        output_files[lang] = output_file
        print(f"Language '{lang}': {len(messages)} records written to {output_file}")

    print(f"\nTotal: {sum(len(m) for m in messages_by_lang.values())} records across {len(messages_by_lang)} languages")
    return output_files


if __name__ == "__main__":
    input_csv = "validation.csv"
    convert_csv_to_json_by_language(input_csv, output_prefix="jisaw_dataset")
