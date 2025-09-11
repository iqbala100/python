import os
import re
import shutil
import yaml
from deepdiff import DeepDiff

# === Configure paths (use raw strings r"" on Windows) ===
input_dir              = r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG1\.harness\services"
matching_common_folder = r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\yaml_matching_common"

# === Behavior toggles ===
# "exact": copy files only when YAMLs are identical (DeepDiff == empty, ignore_order=True)
# "threshold": treat files as matching if their flattened (path,value) similarity >= SIMILARITY_THRESHOLD
MATCH_MODE = "exact"        # choose: "exact" or "threshold"
SIMILARITY_THRESHOLD = 0.90 # used only when MATCH_MODE == "threshold"

# Compare only files with the same base name across folders (reduces pair count)
MATCH_BY_NAME = False

# Preserve original subfolder structure inside the common folder
PRESERVE_STRUCTURE = False

# Empty destination folder before copying (keeps the folder itself)
CLEAN_DEST = False


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
        yield (".".join(path), obj)

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

def copy_to_common(src: str, input_root: str, common_dir: str, preserve_structure: bool):
    rel = os.path.relpath(src, input_root)
    if preserve_structure:
        dst = os.path.join(common_dir, rel)
        ensure_dir(os.path.dirname(dst))
    else:
        dst = os.path.join(common_dir, sane_name(rel))
        root, ext = os.path.splitext(dst)
        if not ext:
            dst = root + ".yaml"
    shutil.copy2(src, dst)
    return dst


# ---------- main ----------

def main():
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

    yaml_files = list_yaml_files_recursively(input_dir)
    if len(yaml_files) < 2:
        print("Need at least two YAML files under input_dir to compare.")
        return

    # Build comparison pairs
    if MATCH_BY_NAME:
        from collections import defaultdict
        groups = defaultdict(list)
        for abs_path, rel in yaml_files:
            groups[os.path.basename(rel)].append((abs_path, rel))
        pairs = []
        for _, items in groups.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    pairs.append((items[i], items[j]))
    else:
        pairs = []
        for i in range(len(yaml_files)):
            for j in range(i + 1, len(yaml_files)):
                pairs.append((yaml_files[i], yaml_files[j]))

    files_to_copy = set()
    total_pairs = 0
    matched_pairs = 0

    for (path1, rel1), (path2, rel2) in pairs:
        total_pairs += 1
        try:
            data1 = load_first_yaml_doc(path1) or {}
            data2 = load_first_yaml_doc(path2) or {}
        except Exception:
            # Skip unreadable YAMLs
            continue

        diff = DeepDiff(data1, data2, ignore_order=True)

        if MATCH_MODE == "exact":
            is_match = (not diff)  # exact structural match
        else:
            flat1 = dict(flatten(data1))
            flat2 = dict(flatten(data2))
            sim = similarity(flat1, flat2)
            is_match = (sim >= SIMILARITY_THRESHOLD)

        if is_match:
            matched_pairs += 1
            files_to_copy.add(path1)
            files_to_copy.add(path2)

    # Copy unique matched files to the common folder
    copied = 0
    for src in sorted(files_to_copy):
        copy_to_common(src, input_dir, matching_common_folder, PRESERVE_STRUCTURE)
        copied += 1

    print(f"Done.\n  Compared pairs: {total_pairs}")
    print(f"  Matching pairs: {matched_pairs}")
    print(f"  Unique files copied to common folder: {copied} -> {matching_common_folder}")

if __name__ == "__main__":
    main()
