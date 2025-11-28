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
- **Collection**: Verify entire collection as one unit (generates combined ISCC)

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

### Scenario 1: Quick Collection Integrity Check
**Goal**: Quickly verify if an entire collection has changed

```
Workflow:
1. Generate ISCC (combined mode) → Input: Reference collection (100 files)
                                   Output: Single ISCC hash

2. [Transfer/storage/time passes]

3. Verify ISCC → Input: New collection (100 files) + reference hash
                 Output: "Status: OK" or "Status: FAILED"
```

**Result**: You know instantly if the collection as a whole has changed (but not which specific files)

---

### Scenario 2: Dataset Similarity When Verification Fails
**Goal**: Check if a large dataset matches reference exactly, and if not, how similar it is

**Why this is important**: Sometimes you receive a dataset that's been modified but want to know if it's still what you expect it to be or how much it differs from the reference.

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

### Scenario 3: Duplicate Detection in Large Dataset
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

2. Generate ISCC (combined mode) → Create hash for collection

3. Verify ISCC → Compare against expected reference
                 ↓
    PASS: Continue workflow
    FAIL: → 4. Compare similarity (two collections)
              ↓
              Report similarity score (~08 = very similar, ~48 = very different)
              Decide: Accept if similar enough, or reject and investigate
```

**Result**: Automated QC catches data integrity issues and helps determine if differences are acceptable

---

## More Information

- ISCC specification: https://iscc.codes/
- ISCC-SUM GitHub: https://github.com/iscc/iscc-sum
