import csv
import json


def convert_csv_to_json(input_file, output_file):
    """
    Convert CSV with toxicity data to JSON format.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output JSON file
    """
    result = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract toxicity flags
            toxicity_flags = [
                int(row['toxic']),
                int(row['severe_toxic']),
                int(row['obscene']),
                int(row['threat']),
                int(row['insult']),
                int(row['identity_hate'])
            ]

            # Check if any flag is raised
            is_toxic = any(flag == 1 for flag in toxicity_flags)

            # Calculate score: sum of flags divided by total number of flags
            score = sum(toxicity_flags) / len(toxicity_flags)

            # Create JSON object
            json_obj = {
                "message": row['comment_text'],
                "score": score,
                "toxic": is_toxic
            }

            result.append(json_obj)

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Conversion complete! {len(result)} records written to {output_file}")
    return result


# Usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_csv = "validation.csv"
    output_json = "toxicity_data.json"

    convert_csv_to_json(input_csv, output_json)
