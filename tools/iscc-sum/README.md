# ISCC-SUM Tools for Galaxy

A suite of Galaxy tools for generating, verifying, and comparing ISCC (International Standard Content Code) hashes. ISCC-SUM provides content-derived identifiers for file integrity verification and similarity detection.

## Tools Overview

### 1. Generate ISCC hash (`iscc_sum.xml`)
**Purpose**: Create reference ISCC hashes for files

**Modes**:
- **Single file**: Generate one ISCC hash
- **Collection (individual)**: Generate hash per file with element identifiers
- **Collection (combined)**: Generate single hash for entire collection

**Use when**: You need to create reference hashes for later verification or comparison

---

### 2. Verify ISCC hash (`iscc_verify.xml`)
**Purpose**: Exact match verification - check if files are identical

**Modes**:
- **Single file**: Verify one file against expected hash
- **Collection**: Verify all files in collection against reference

**Use when**: You need to confirm files are EXACTLY the same (bit-for-bit)

**Note**: Even minor modifications will cause verification to FAIL

---

### 3. Compare ISCC similarity (`iscc_similarity.xml`)
**Purpose**: Content similarity detection - find related/modified files

**Modes**:
- **Two files**: Compare two specific files
- **Collection**: Find all similar files within a collection
- **Two collections**: Compare two entire collections as units (generates combined ISCC for each)

**Use when**: You want to detect:
- Near-duplicates (minor edits, format changes)
- Different versions of same content
- Related files (cropped images, edited documents)
- Whether two entire datasets are similar (even if not exactly identical)

**Key feature**: Works especially well on **large datasets** - ISCC-SUM is optimized for detecting similarity across many files efficiently

---

## Usage Scenarios

### Scenario 1: Data Integrity Verification
**Goal**: Ensure received dataset matches reference exactly

```
Workflow:
1. Generate ISCC → Input: Reference collection (100 files)
                  Output: reference_hashes.txt

2. [Transfer/storage/time passes]

3. Verify ISCC → Input: New collection (100 files) + reference_hashes.txt
                 Output: "Passed: 95, Failed: 5"
```

**Result**: You know exactly which 5 files were modified/corrupted

---

### Scenario 2: Dataset Similarity When Verification Fails
**Goal**: Check if a large dataset matches reference exactly, and if not, how similar it is

**Why this is important**: Sometimes you receive a dataset that's been modified but want to know if it's still usable or how much it differs from the reference.

```
Workflow:
1. Generate ISCC (combined mode) → Input: Reference collection (100 files)
                                   Output: Single ISCC hash

2. [Transfer/storage/time passes - dataset may have changed]

3. Verify ISCC → Input: New collection (100 files) + reference hash
                 Output: "Status: FAILED - Hashes do not match"

4. Compare similarity (two collections) → Input: Reference collection + New collection
                                          Threshold: 12
                                          Output: "Similarity: ~08 (Very similar, minor changes)"
```

**Result**:
- You know the dataset is NOT identical (verification failed)
- You know it's still very similar (Hamming distance = 8)
- You can make an informed decision: Is it similar enough to use? Should you investigate the differences?

**Use cases**:
- Receiving scientific datasets from collaborators
- Verifying backups that may have compression/format changes
- Checking if processed data still matches original closely enough
- Quality control where "close enough" is acceptable

---

### Scenario 3: Finding Modified Files
**Goal**: Identify which files changed and by how much

```
Workflow:
1. Verify ISCC → Shows 5 files failed verification

2. Compare similarity → Input: Collection with all 200 files (reference + new)
                        Output: Similarity report showing:
                        - file_023: ~03 (nearly identical, minor edit)
                        - file_045: ~48 (completely different)
                        - file_067: ~12 (moderate changes)
```

**Result**: You can prioritize which files need attention based on similarity scores

---

### Scenario 4: Duplicate Detection in Large Dataset
**Goal**: Find duplicate and near-duplicate files in large collection

**Why ISCC-SUM excels here**: Traditional hash functions (MD5, SHA) only detect exact duplicates. ISCC-SUM detects **content similarity**, making it ideal for:
- Image collections (same image with edits)
- Document repositories (different versions, formats)
- Genomic data (similar sequences with variations)

```
Workflow:
1. Compare similarity → Input: Large collection (1000+ files)
                        Threshold: 12 (configurable)
                        Output: Groups of similar files

Example output:
  reference_image_001.png
    ~00 duplicate_001.jpg (exact duplicate, different format)
    ~08 edited_001.png (minor edits)

  document_v1.txt
    ~05 document_v2.txt (very similar)
    ~12 document_draft.txt (moderate differences)
```

**Result**: Identify redundant files, track versions, find related content

---

### Scenario 4: Quality Control Pipeline
**Goal**: Automated verification in data processing pipeline

```
Galaxy Workflow:
1. [Data arrives] → Collection of files

2. Generate ISCC (individual mode) → Create hashes for all files

3. Verify ISCC → Compare against expected reference
                 ↓
    PASS: Continue workflow
    FAIL: → 4. Compare similarity
              ↓
              Report discrepancies and similarity scores
              Stop or flag for manual review
```

**Result**: Automated QC catches data integrity issues early

---

### Scenario 5: Collection Completeness Check
**Goal**: Verify you have all expected files

```
Workflow:
1. Generate ISCC → Input: Expected complete collection
                  Output: complete_reference.txt (100 files)

2. Verify ISCC → Input: Received collection (95 files) + complete_reference.txt
                 Output:
                 - Passed: 92
                 - Failed: 3
                 - NOT FOUND in reference: 0
                 - Missing from collection: 5
```

**Result**: Immediately identify missing or modified files

---


## Combined Mode Use Case

The "combined mode" in Generate ISCC creates a single hash for entire collection:

```
Use when:
- Verifying a multi-file dataset as a unit
- Creating checksum for entire directory
- Detecting if ANY file in collection changed

Example:
Reference collection: 10 files → Combined ISCC: K4AXYZ...
Later check: 10 files → Combined ISCC: K4AXYZ... (match!)

If even ONE file changed → Combined ISCC will be different
```

**Note**: This doesn't tell you WHICH file changed, only that collection changed. Use individual mode for per-file tracking.


---

## More Information

- ISCC specification: https://iscc.codes/
- ISCC-SUM GitHub: https://github.com/iscc/iscc-sum
