import argparse, os, re
from collections import defaultdict

"""Check duplicated bib items by their key across multiple files.
1. Convert all bib items key to lowercase.
2. Check duplicated keys, and their corresponding file and line number.
"""

def lower_n_check_duplicates(bib_files):
    key_pattern = re.compile(r'@\w+\{(\w+),')  # Regex to match bib item keys
    all_keys = defaultdict(list)  # Dictionary to store keys, filenames, and line numbers

    for bib_file in bib_files:
        with open(bib_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # convert to lowercaes & write back in-place
        # NOTE no back-up, will over-write original file
        with open(bib_file, 'w', encoding='utf-8') as file:
            for line_number, line in enumerate(lines, start=1):
                match = key_pattern.search(line)
                if match:
                    original_key = match.group(1)
                    lower_key = original_key.lower()

                    # Record the key with filename and line number
                    all_keys[lower_key].append((bib_file, line_number))

                    # Replace the key in the line
                    line = line.replace(original_key, lower_key, 1)

                file.write(line)

    # Check for duplicates
    duplicates = {key: locations for key, locations in all_keys.items() if len(locations) > 1}

    if duplicates:
        for key, locations in duplicates.items():
            print(key)
            n = len(locations)
            for i, (filename, lineno) in enumerate(locations):
                if i + 1 == n:
                    print("`-: {} : {}".format(filename, lineno))
                else:
                    print("|-: {} : {}".format(filename, lineno))


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib_files", nargs='+', help="List of .bib files to process")
    args = parser.parse_args()

    lower_n_check_duplicates(args.bib_files)
