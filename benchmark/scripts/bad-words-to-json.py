import os
import json


def txt_files_to_json(folder_path, output_file='output.json'):
    """
    Reads all .txt files from a folder and creates a single JSON file.

    Args:
        folder_path (str): Path to the folder containing txt files
        output_file (str): Name of the output JSON file (default: 'output.json')
    """
    result = {}

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in '{folder_path}'")
        return

    # Process each txt file
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)

        # Read the file and split into lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.rstrip('\n') for line in file.readlines()]

        # Use filename without extension as the key
        file_key = os.path.splitext(filename)[0]
        result[file_key] = lines

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)

    print(f"Successfully created '{output_file}' with {len(txt_files)} file(s)")
    print(f"Files processed: {', '.join(txt_files)}")


if __name__ == "__main__":
    # Example usage - modify the folder path as needed
    folder_path = "./input"  # Current directory, change to your folder path
    output_file = "output.json"

    txt_files_to_json(folder_path, output_file)
