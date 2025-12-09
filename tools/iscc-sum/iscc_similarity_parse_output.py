#!/usr/bin/env python
"""
Parse ISCC similarity output into tabular format with unique identifiers.

Input format (from iscc-sum --similar):
    ISCC:K4AOMG... *file1.txt
      ~08 ISCC:K4AOMG... *file2.txt
      ~10 ISCC:K4AOMG... *file3.txt
    ISCC:K4AGSPO... *file4.txt

Output format (tabular with 7 columns, bidirectional):
    file_id      filename    iscc_hash    match_id     match_filename    match_iscc_hash    distance
    dataset_123  file1.txt   K4AOMG...    dataset_124  file2.txt         K4AOMG...          8
    dataset_124  file2.txt   K4AOMG...    dataset_123  file1.txt         K4AOMG...          8
    dataset_125  file4.txt   K4AGSPO...                                                      -1
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


def load_id_mapping(mapping_file):
    """Load filename to element_identifier mapping.

    Returns: dict mapping cleaned filename -> element_identifier
    """
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                filename, element_id = parts
                # Clean the filename the same way as in parse
                cleaned = clean_filename(filename)
                mapping[cleaned] = element_id
    return mapping


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
        help='Raw similarity output from iscc-sum --similar'
    )
    parser.add_argument(
        'id_mapping',
        help='TSV file mapping filenames to element identifiers'
    )
    parser.add_argument(
        'output_file',
        help='Tabular output file'
    )
    args = parser.parse_args()

    # Load ID mapping
    id_map = load_id_mapping(args.id_mapping)

    # Parse similarity output
    file_hashes = {}  # filename -> hash mapping
    matches = []  # List of (file1, hash1, file2, hash2, distance)
    current_ref = None
    current_hash = None

    with open(args.similarity_raw, 'r') as f:
        for line in f:
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

    # Write output with identifiers
    with open(args.output_file, 'w') as out:
        # Write header (7 columns)
        out.write("file_id\tfilename\tiscc_hash\tmatch_id\tmatch_filename\tmatch_iscc_hash\tdistance\n")

        # Track which files have matches
        files_with_matches = set()

        # Write similarity matches in both directions
        for file1, hash1, file2, hash2, distance in matches:
            # Get element identifiers
            file1_id = id_map.get(file1, file1)  # Fallback to filename if not found
            file2_id = id_map.get(file2, file2)

            # Write A -> B
            out.write(f"{file1_id}\t{file1}\t{hash1}\t{file2_id}\t{file2}\t{hash2}\t{distance}\n")
            # Write B -> A (bidirectional)
            out.write(f"{file2_id}\t{file2}\t{hash2}\t{file1_id}\t{file1}\t{hash1}\t{distance}\n")

            files_with_matches.add(file1)
            files_with_matches.add(file2)

        # Write files with no matches (distance = -1, empty match columns)
        for filename in sorted(file_hashes.keys()):
            if filename not in files_with_matches:
                file_id = id_map.get(filename, filename)
                hash_val = file_hashes[filename]
                out.write(f"{file_id}\t{filename}\t{hash_val}\t\t\t\t-1\n")


if __name__ == '__main__':
    main()
