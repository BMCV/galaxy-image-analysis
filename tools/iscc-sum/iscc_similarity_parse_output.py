#!/usr/bin/env python
"""
Parse ISCC similarity output into tabular format.

Input format (from iscc-sum --similar):
    ISCC:K4AOMG... *file1.txt
      ~08 ISCC:K4AOMG... *file2.txt
      ~10 ISCC:K4AOMG... *file3.txt
    ISCC:K4AGSPO... *file4.txt

Output format (tabular with 5 columns):
    filename    iscc_hash    match_filename    match_iscc_hash    distance
    file1.txt   K4AOMG...    file2.txt         K4AOMG...          8
    file1.txt   K4AOMG...    file3.txt         K4AOMG...          10
    file4.txt   K4AGSPO...                                        -1
"""
import argparse


def clean_filename(filename):
    """Remove directory prefix and numeric suffix from filename."""
    # Remove 'input_files/' prefix if present
    if filename.startswith('input_files/'):
        filename = filename[len('input_files/'):]

    # Remove trailing '_N' suffix (where N is a digit)
    # e.g., 'test1.png_0' -> 'test1.png'
    if '_' in filename:
        parts = filename.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            filename = parts[0]

    return filename


def parse_iscc_line(line):
    """Parse ISCC line and extract hash and filename.

    Format: "ISCC:HASH *filename" or "  ~NN ISCC:HASH *filename"
    Returns: (hash, filename) or (None, None) if parse fails
    """
    # Find the * separator
    if ' *' not in line:
        return None, None

    # Split on ' *' to get hash part and filename
    parts = line.split(' *', 1)
    hash_part = parts[0].strip()
    filename = clean_filename(parts[1].strip())

    # Extract hash (after 'ISCC:')
    if 'ISCC:' in hash_part:
        hash_code = hash_part.split('ISCC:', 1)[1].strip()
    else:
        hash_code = ''

    return hash_code, filename


def main():
    parser = argparse.ArgumentParser(
        description='Parse ISCC similarity output into tabular format'
    )
    parser.add_argument(
        'similarity_raw',
        type=argparse.FileType('r'),
        help='Raw similarity output from iscc-sum --similar'
    )
    parser.add_argument(
        'output_file',
        type=argparse.FileType('w'),
        help='Tabular output file'
    )
    args = parser.parse_args()

    # Parse similarity output
    file_hashes = {}  # filename -> hash mapping
    matches = []  # List of (file1, hash1, file2, hash2, distance)
    current_ref = None
    current_hash = None

    for line in args.similarity_raw:
        line = line.rstrip()
        if not line:
            continue

        if line.startswith('ISCC:'):
            # Reference file: "ISCC:HASH *filename"
            hash_code, filename = parse_iscc_line(line)
            if hash_code and filename:
                current_ref = filename
                current_hash = hash_code
                file_hashes[filename] = hash_code

        elif line.startswith(' ') and current_ref:
            # Similar file: "  ~NN ISCC:HASH *filename"
            parts = line.strip().split(None, 1)  # Split on first whitespace
            if len(parts) == 2:
                dist_str = parts[0].replace('~', '')
                distance = int(dist_str)

                # Parse the rest of the line for ISCC and filename
                hash_code, filename = parse_iscc_line(parts[1])

                if hash_code and filename:
                    matches.append((current_ref, current_hash, filename, hash_code, distance))
                    file_hashes[filename] = hash_code

    # Write header (5 columns)
    args.output_file.write("filename\tiscc_hash\tmatch_filename\tmatch_iscc_hash\tdistance\n")

    # Track which files have matches
    files_with_matches = set()
    written_pairs = set()

    # Write similarity matches (deduplicated)
    for file1, hash1, file2, hash2, distance in matches:
        # Avoid duplicate pairs (A-B and B-A)
        pair = tuple(sorted([file1, file2]))
        if pair not in written_pairs:
            args.output_file.write(f"{file1}\t{hash1}\t{file2}\t{hash2}\t{distance}\n")
            written_pairs.add(pair)
            files_with_matches.add(file1)
            files_with_matches.add(file2)

    # Write files with no matches (distance = -1, empty match columns)
    for filename in sorted(file_hashes.keys()):
        if filename not in files_with_matches:
            hash_val = file_hashes[filename]
            args.output_file.write(f"{filename}\t{hash_val}\t\t\t-1\n")

    args.output_file.close()
    args.similarity_raw.close()


if __name__ == '__main__':
    main()
