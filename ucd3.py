import os
import re
import shutil
import hashlib
import yaml
from deepdiff import DeepDiff

# === Configure paths (use raw strings r"" on Windows) ===
input_dirs = [
    r"C:\path\to\envA\.harness\services",
    r"C:\path\to\envB\.harness\services",
    # add more roots as needed...
]
matching_common_folder = r"C:\path\to\yaml_matching_common"

# === Behavior toggles ===
# "exact": copy files only when YAMLs are identical (DeepDiff == empty, ignore_order=True)
# "threshold": treat files as matching if their flattened (path,value) similarity >= SIMILARITY_THRESHOLD
MATCH_MODE = "exact"         # choose: "exact" or "threshold"
SIMILARITY_THRESHOLD = 0.90  # used only when MATCH_MODE == "threshold"

# Compare only files with the same base name across folders (reduces pair count)
MATCH_BY_NAME = False

# Preserve original subfolder structure (grouped by root label) inside the common folder.
# If False, the destination is flat and file names are path-encoded with the root label.
PRESERVE_STRUCTURE = False

# Empty destination folder before copying (keeps the folder itself)
CLEAN_DEST = False

# Compare files within the same input root as well (default compares only across different roots)
INCLUDE_INTRA_ROOT = False

# Write a single YAML with pairwise common values for matched pairs
WRITE_COMMON_VALUES_REPORT = True
COMMON_VALUES_REPORT_NAME = "common_values_report.yaml"
# Limit the number of common entries stored per pair (to keep report manageable)
MAX_COMMON_VALUES_PER_PAIR = 200


# ---------- helpers ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_yaml_files_recursively(root_dir: str):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith((".yaml", ".yml")):
                full = os.path.join(root, name)
                rel  = os.path.relpath(full, root_dir)
                files.append((full, rel))  # (absolute_path, relative_path)
    return files

def load_first_yaml_doc(path: str):
    # Compare the first document if a file contains multiple docs separated by '---'
    with open(path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
        if not docs:
            return {}
        return docs[0] if docs[0] is not None else {}

def flatten(obj, path=()):
    """
    Flatten dicts/lists into {"a.b[0].c": value, ...} for path-wise equality & similarity.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, path + (f"[{i}]",))
    else:
        key = ".".join(path) if path else "<root>"
        yield (key, obj)

def sane_name(s: str) -> str:
    """
    Safe filename from a relative path (keep dot, dash, underscore).
    """
    s = s.replace(os.sep, "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def similarity(flat1: dict, flat2: dict) -> float:
    """
    Jaccard-like similarity on exact (path,value) matches.
    score = |intersection| / |union|
    """
    set1 = set(flat1.items())
    set2 = set(flat2.items())
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    inter = set1 & set2
    return len(inter) / len(union) if union else 0.0

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def copy_to_common(src_abs: str, src_rel: str, root_label: str, common_dir: str, preserve_structure: bool):
    """
    Copy file into common_dir.
    - If preserve_structure: common_dir / root_label / src_rel
    - Else: common_dir / (root_label__sanitized_rel[.yaml])
    """
    if preserve_structure:
        dst = os.path.join(common_dir, root_label, src_rel)
        ensure_dir(os.path.dirname(dst))
    else:
        dst = os.path.join(common_dir, f"{root_label}__{sane_name(src_rel)}")
        root, ext = os.path.splitext(dst)
        if not ext:
            dst = root + ".yaml"
    shutil.copy2(src_abs, dst)
    return dst


# ---------- main ----------

def main():
    if not input_dirs or len(input_dirs) < 2:
        print("Provide at least two input directories in 'input_dirs'.")
        return

    ensure_dir(matching_common_folder)

    if CLEAN_DEST and os.path.isdir(matching_common_folder):
        # Remove everything inside common_dir (but keep the folder)
        for root, dirs, files in os.walk(matching_common_folder, topdown=False):
            for name in files:
                try:
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass

    # Gather files across all roots: (root_idx, root_label, root_path, abs_path, rel_path)
    all_files = []
    for idx, root in enumerate(input_dirs):
        label = os.path.basename(os.path.normpath(root)) or f"root{idx+1}"
        for abs_p, rel_p in list_yaml_files_recursively(root):
            all_files.append((idx, label, root, abs_p, rel_p))

    if len(all_files) < 2:
        print("Need at least two YAML files across the given input directories.")
        return

    # Build comparison pairs
    if MATCH_BY_NAME:
        from collections import defaultdict
        groups = defaultdict(list)
        for entry in all_files:
            idx, label, root, abs_p, rel_p = entry
            groups[os.path.basename(rel_p)].append(entry)
        pairs = []
        for _, items in groups.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a = items[i]; b = items[j]
                    if not INCLUDE_INTRA_ROOT and a[0] == b[0]:
                        continue
                    pairs.append((a, b))
    else:
        pairs = []
        for i in range(len(all_files)):
            for j in range(i + 1, len(all_files)):
                a = all_files[i]; b = all_files[j]
                if not INCLUDE_INTRA_ROOT and a[0] == b[0]:
                    continue
                pairs.append((a, b))

    if not pairs:
        print("No file pairs to compare (check MATCH_BY_NAME / INCLUDE_INTRA_ROOT).")
        return

    # Compare & collect matches + common values
    matched_paths = set()     # abs paths that matched at least one file
    index_by_abs = {e[3]: e for e in all_files}
    common_values_report = []  # list of dicts per matched pair

    total_pairs = 0
    matched_pairs = 0

    for A, B in pairs:
        total_pairs += 1
        idxA, labelA, rootA, pathA, relA = A
        idxB, labelB, rootB, pathB, relB = B
        try:
            d1 = load_first_yaml_doc(pathA) or {}
            d2 = load_first_yaml_doc(pathB) or {}
        except Exception:
            continue  # skip unreadable YAMLs

        diff = DeepDiff(d1, d2, ignore_order=True)

        sim = None
        if MATCH_MODE == "exact":
            is_match = (not diff)
        else:
            flat1 = dict(flatten(d1))
            flat2 = dict(flatten(d2))
            sim = similarity(flat1, flat2)
            is_match = (sim >= SIMILARITY_THRESHOLD)

        if is_match:
            matched_pairs += 1
            matched_paths.add(pathA)
            matched_paths.add(pathB)

            # Compute common values (by full key path)
            flat1 = dict(flatten(d1))
            flat2 = dict(flatten(d2))
            commons = [{"path": k, "value": flat1[k]}
                       for k in sorted(set(flat1.keys()) & set(flat2.keys()))
                       if flat1[k] == flat2[k]]

            # Trim to keep the report small (configurable)
            commons_trimmed = commons[:MAX_COMMON_VALUES_PER_PAIR]

            if WRITE_COMMON_VALUES_REPORT:
                common_values_report.append({
                    "fileA": {"root": labelA, "rel": relA},
                    "fileB": {"root": labelB, "rel": relB},
                    "mode": MATCH_MODE,
                    "similarity": None if sim is None else round(sim, 4),
                    "common_count": len(commons),
                    "common_values_sampled": commons_trimmed
                })

    # Copy unique matched files to the common folder (dedupe by file content)
    copied = 0
    seen_hashes = set()
    for abs_p in sorted(matched_paths):
        idx, label, root, _, rel = index_by_abs[abs_p]
        try:
            h = file_sha256(abs_p)
        except Exception:
            h = f"path::{abs_p}"
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        copy_to_common(abs_p, rel, label, matching_common_folder, PRESERVE_STRUCTURE)
        copied += 1

    # Write common values report (single YAML)
    if WRITE_COMMON_VALUES_REPORT:
        try:
            report = {
                "config": {
                    "match_mode": MATCH_MODE,
                    "similarity_threshold": SIMILARITY_THRESHOLD if MATCH_MODE == "threshold" else None,
                    "match_by_name": MATCH_BY_NAME,
                    "preserve_structure": PRESERVE_STRUCTURE,
                    "include_intra_root": INCLUDE_INTRA_ROOT,
                    "max_common_values_per_pair": MAX_COMMON_VALUES_PER_PAIR,
                },
                "stats": {
                    "input_roots": len(input_dirs),
                    "yaml_files_discovered": len(all_files),
                    "pairs_compared": total_pairs,
                    "matching_pairs": matched_pairs,
                    "unique_files_copied": copied
                },
                "pairs": common_values_report
            }
            report_path = os.path.join(matching_common_folder, COMMON_VALUES_REPORT_NAME)
            with open(report_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
            print(f"Common values report (YAML) written to: {report_path}")
        except Exception as e:
            print(f"Failed to write YAML report: {e}")

    print("Done.")
    print(f"  Input roots           : {len(input_dirs)}")
    print(f"  YAML files discovered : {len(all_files)}")
    print(f"  Pairs compared        : {total_pairs}")
    print(f"  Matching pairs        : {matched_pairs}")
    print(f"  Unique files copied   : {copied} -> {matching_common_folder}")


if __name__ == "__main__":
    main()
