#!/usr/bin/env python
"""
Parse ISCC similarity output into tabular format with unique identifiers.

Input format (from iscc-sum --similar):
    ISCC:K4AOMG... *file1.txt
      ~08 ISCC:K4AOMG... *file2.txt
      ~10 ISCC:K4AOMG... *file3.txt
    ISCC:K4AGSPO... *file4.txt

Output format (tabular with 7 columns, bidirectional):
    file_id      filename    iscc_code    match_id     match_filename    match_iscc_hash    distance
    23  file1.txt   K4AOMG...    24  file2.txt         K4AOMG...          8
    24  file2.txt   K4AOMG...    23  file1.txt         K4AOMG...          8
    25  file4.txt   K4AGSPO...                                                      -1
"""
import argparse


def clean_filename(filename):
    """Remove directory prefix from filename."""
    # Remove 'input_files/' prefix if present
    if filename.startswith('input_files/'):
        filename = filename[len('input_files/'):]

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
    """Parse ISCC line and extract code and filename.

    Format: "ISCC:CODE *filename" or "  ~NN ISCC:CODE *filename"
    Returns: (code, filename) or (None, None) if parse fails
    """
    # Find the * separator
    if ' *' not in line:
        return None, None

    # Split on ' *' to get code part and filename
    parts = line.split(' *', 1)
    code_part = parts[0].strip()
    filename = clean_filename(parts[1].strip())

    # Extract CODE (after 'ISCC:')
    if 'ISCC:' in code_part:
        code = code_part.split('ISCC:', 1)[1].strip()
    else:
        code = ''

    return code, filename


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
    file_codes = {}  # filename -> code mapping
    matches = []  # List of (file1, code1, file2, code2, distance)
    current_ref = None
    current_code = None

    with open(args.similarity_raw, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith('ISCC:'):
                # Reference file: "ISCC:CODE *filename"
                code, filename = parse_iscc_line(line)
                if code and filename:
                    current_ref = filename
                    current_code = code
                    file_codes[filename] = code

            elif line.startswith(' ') and current_ref:
                # Similar file: "  ~NN ISCC:CODE *filename"
                parts = line.strip().split(None, 1)  # Split on first whitespace
                if len(parts) == 2:
                    dist_str = parts[0].replace('~', '')
                    distance = int(dist_str)

                    # Parse the rest of the line for ISCC and filename
                    code, filename = parse_iscc_line(parts[1])

                    if code and filename:
                        matches.append((current_ref, current_code, filename, code, distance))
                        file_codes[filename] = code
    # Write output with identifiers
    with open(args.output_file, 'w') as out:
        # Write header (7 columns)
        out.write("file_id\tfilename\tiscc_code\tmatch_id\tmatch_filename\tmatch_iscc_code\tdistance\n")

        # Track which files have matches
        files_with_matches = set()

        # Write similarity matches in both directions
        for file1, code1, file2, code2, distance in matches:
            # Get element identifiers
            file1_name = id_map[file1]
            file2_name = id_map[file2]
            file1_id = str.split(file1, '_', 1)[0]  # Extract ID from filename
            file2_id = str.split(file2, '_', 1)[0]  # Extract ID from filename

            # Write A -> B (file_id is the numeric ID, filename is the element_identifier)
            out.write(f"{file1_id}\t{file1_name}\t{code1}\t{file2_id}\t{file2_name}\t{code2}\t{distance}\n")
            # Write B -> A (bidirectional)
            out.write(f"{file2_id}\t{file2_name}\t{code2}\t{file1_id}\t{file1_name}\t{code1}\t{distance}\n")

            files_with_matches.add(file1)
            files_with_matches.add(file2)

        # Write files with no matches (distance = -1, empty match columns)
        for filename in sorted(file_codes.keys()):
            if filename not in files_with_matches:
                file_id = str.split(filename, '_', 1)[0]  # Extract ID from filename
                element_name = id_map[filename]
                code_val = file_codes[filename]
                out.write(f"{file_id}\t{element_name}\t{code_val}\t\t\t\t-1\n")


if __name__ == '__main__':
    main()
