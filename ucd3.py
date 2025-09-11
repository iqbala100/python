import os
import re
import shutil
import hashlib
import yaml
from typing import Any, Dict, List, Tuple, Optional

# ========== CONFIG ==========
# Multiple input roots to scan recursively
input_dirs = [
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG1\.harness\services",
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG2\.harness\services",
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG3\.harness\services",
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG4\.harness\services",
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG5\.harness\services",
    r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\harness_out_RG6\.harness\services",
    # add more roots if needed...
]

# Single destination for matched files + report
matching_common_folder = r"C:\Users\iqahmad\Desktop\RFP\GitHubRepo\ucd-to-harness1\yaml_matching_common"

# --- Metadata match rules (PRIMARY) ---
REQUIRE_SAME_NAME = True          # require YAML 'name' equality
REQUIRE_SAME_IDENTIFIER = True    # require YAML 'identifier' equality

# Tags matching mode:
#   'any'     -> at least 1 tag in common
#   'min'     -> at least TAGS_INTERSECT_MIN in common
#   'subset'  -> tags of one file ⊆ tags of the other
#   'exact'   -> sets equal (ignoring order)
TAGS_MATCH_MODE = "any"
TAGS_INTERSECT_MIN = 1            # used when TAGS_MATCH_MODE == 'min'
TAGS_REQUIRED = False             # if True, both files must have non-empty tags to be considered

# Normalize comparisons
CASE_INSENSITIVE_COMPARE = True   # lowercases name, identifier, tag keys/values

# Where to look for fields inside YAML (first hit wins)
FIELD_PATHS = {
    "name":       ["service.name", "name", "metadata.name"],
    "identifier": ["service.identifier", "identifier", "metadata.identifier", "id"],
    "tags":       ["service.tags", "tags", "metadata.tags"]
}

# --- Optional content check (SECONDARY) ---
# After metadata matches, optionally also require content similarity:
ENABLE_CONTENT_CHECK = False      # set True to also check content with DeepDiff
CONTENT_MODE = "exact"            # 'exact' or 'threshold'
SIMILARITY_THRESHOLD = 0.90       # used when CONTENT_MODE == 'threshold'

# Compare only files with the same base filename across roots (pre-filter)
MATCH_BY_FILENAME = False

# Mirror original subfolders grouped by root label in the common folder; else use flat, safe names
PRESERVE_STRUCTURE = False

# Clear destination before copying
CLEAN_DEST = False

# Also compare files within the same input root (default False = cross-root only)
INCLUDE_INTRA_ROOT = False

# YAML report name
REPORT_NAME = "matches_report.yaml"

# Limit common-keys sample stored per pair to keep report small
MAX_COMMON_SAMPLED = 200
# ========== END CONFIG ==========

# ---- Optional DeepDiff usage (only if ENABLE_CONTENT_CHECK is True) ----
try:
    from deepdiff import DeepDiff  # noqa
except Exception:
    if ENABLE_CONTENT_CHECK:
        raise RuntimeError("DeepDiff required: pip install deepdiff")


# ================= helpers =================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_yaml_files_recursively(root_dir: str) -> List[Tuple[str, str]]:
    files = []
    for root, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith((".yaml", ".yml")):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, root_dir)
                files.append((full, rel))
    return files

def load_first_yaml_doc(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
        if not docs:
            return {}
        return docs[0] if docs[0] is not None else {}

def get_by_path(obj: Any, dotted: str) -> Optional[Any]:
    """
    Traverse obj using dotted path like 'service.tags' or 'metadata.name'.
    Supports simple dict keys; does not index lists for these fields.
    """
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def find_first(obj: Any, paths: List[str]) -> Optional[Any]:
    for p in paths:
        v = get_by_path(obj, p)
        if v is not None:
            return v
    return None

def norm_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip()
    return v.lower() if CASE_INSENSITIVE_COMPARE else v

def normalize_tags(raw: Any) -> List[str]:
    """
    Return a list of normalized tags as 'key=value' or single token.
    Accepts:
      - dict: {"k":"v", "team":"EDO"}
      - list[str]: ["EDO", "backend"]
      - list[dict]: [{"k":"v"}, {"team":"EDO"}]
    """
    out = []
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            k2 = norm_str(k)
            v2 = "" if v is None else norm_str(v)
            out.append(f"{k2}={v2}")
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                for k, v in item.items():
                    k2 = norm_str(k)
                    v2 = "" if v is None else norm_str(v)
                    out.append(f"{k2}={v2}")
            else:
                out.append(norm_str(item))
    else:
        out.append(norm_str(raw))
    # dedupe while preserving order
    seen = set()
    dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup

def extract_metadata(doc: Any) -> Dict[str, Any]:
    name = find_first(doc, FIELD_PATHS["name"])
    ident = find_first(doc, FIELD_PATHS["identifier"])
    tags = find_first(doc, FIELD_PATHS["tags"])
    return {
        "name": norm_str(name),
        "identifier": norm_str(ident),
        "tags": normalize_tags(tags),
    }

def tags_match(tags_a: List[str], tags_b: List[str]) -> Tuple[bool, Dict[str, Any]]:
    set_a, set_b = set(tags_a), set(tags_b)
    inter = sorted(list(set_a & set_b))
    if TAGS_REQUIRED and (not set_a or not set_b):
        return (False, {"reason": "tags_required_missing", "shared": []})
    if TAGS_MATCH_MODE == "any":
        return (len(inter) >= 1, {"shared": inter})
    if TAGS_MATCH_MODE == "min":
        return (len(inter) >= TAGS_INTERSECT_MIN, {"shared": inter})
    if TAGS_MATCH_MODE == "subset":
        ok = set_a.issubset(set_b) or set_b.issubset(set_a)
        return (ok, {"shared": inter})
    if TAGS_MATCH_MODE == "exact":
        ok = set_a == set_b
        return (ok, {"shared": inter})
    # default fallback = any
    return (len(inter) >= 1, {"shared": inter})

def metadata_match(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    reasons = []
    # name
    if REQUIRE_SAME_NAME:
        if not (meta_a["name"] and meta_b["name"] and meta_a["name"] == meta_b["name"]):
            return (False, {"reason": "name_mismatch", "a": meta_a["name"], "b": meta_b["name"]})
        reasons.append("name_eq")
    # identifier
    if REQUIRE_SAME_IDENTIFIER:
        if not (meta_a["identifier"] and meta_b["identifier"] and meta_a["identifier"] == meta_b["identifier"]):
            return (False, {"reason": "identifier_mismatch", "a": meta_a["identifier"], "b": meta_b["identifier"]})
        reasons.append("identifier_eq")
    # tags
    ok, info = tags_match(meta_a["tags"], meta_b["tags"])
    if not ok:
        info.setdefault("reason", "tags_mismatch")
        return (False, info)
    reasons.append(f"tags_{TAGS_MATCH_MODE}")
    return (True, {"reason": "metadata_match", "shared_tags": info.get("shared", []), "checks": reasons})

def flatten(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, path + (f"[{i}]",))
    else:
        key = ".".join(path) if path else "<root>"
        yield (key, obj)

def similarity(flat1: Dict[str, Any], flat2: Dict[str, Any]) -> float:
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

def sane_name(s: str) -> str:
    s = s.replace(os.sep, "__")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def build_dest(abs_p: str, rel_p: str, root_label: str) -> str:
    if PRESERVE_STRUCTURE:
        dst = os.path.join(matching_common_folder, root_label, rel_p)
        ensure_dir(os.path.dirname(dst))
        return dst
    safe = sane_name(rel_p)
    dst = os.path.join(matching_common_folder, f"{root_label}__{safe}")
    root, ext = os.path.splitext(dst)
    if not ext:
        dst = root + ".yaml"
    return dst

# ================= main =================
def main():
    if not input_dirs or len(input_dirs) < 2:
        print("Provide at least two input directories in 'input_dirs'.")
        return

    ensure_dir(matching_common_folder)
    if CLEAN_DEST:
        for root, dirs, files in os.walk(matching_common_folder, topdown=False):
            for name in files:
                try: os.remove(os.path.join(root, name))
                except Exception: pass
            for name in dirs:
                try: os.rmdir(os.path.join(root, name))
                except Exception: pass

    # Collect files across roots
    all_files: List[Tuple[int, str, str, str, str]] = []  # (root_idx, label, root, abs, rel)
    for idx, root in enumerate(input_dirs):
        label = os.path.basename(os.path.normpath(root)) or f"root{idx+1}"
        for abs_p, rel_p in list_yaml_files_recursively(root):
            all_files.append((idx, label, root, abs_p, rel_p))
    if len(all_files) < 2:
        print("No YAML files found.")
        return

    # Build pairs
    pairs = []
    if MATCH_BY_FILENAME:
        from collections import defaultdict
        groups = defaultdict(list)
        for item in all_files:
            groups[os.path.basename(item[4])].append(item)
        for _, items in groups.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i], items[j]
                    if not INCLUDE_INTRA_ROOT and a[0] == b[0]:
                        continue
                    pairs.append((a, b))
    else:
        for i in range(len(all_files)):
            for j in range(i + 1, len(all_files)):
                a, b = all_files[i], all_files[j]
                if not INCLUDE_INTRA_ROOT and a[0] == b[0]:
                    continue
                pairs.append((a, b))
    if not pairs:
        print("No pairs to compare (check flags).")
        return

    # Prepare index
    index_by_abs = {e[3]: e for e in all_files}

    # Run comparisons with metadata-first match
    matched_paths = set()
    report_pairs = []
    total_pairs = 0

    for A, B in pairs:
        total_pairs += 1
        idxA, labelA, rootA, pathA, relA = A
        idxB, labelB, rootB, pathB, relB = B

        try:
            d1 = load_first_yaml_doc(pathA) or {}
            d2 = load_first_yaml_doc(pathB) or {}
        except Exception:
            continue

        metaA = extract_metadata(d1)
        metaB = extract_metadata(d2)

        # Metadata-based match
        meta_ok, meta_info = metadata_match(metaA, metaB)
        if not meta_ok:
            continue

        content_ok = True
        content_detail = {"mode": "none"}
        if ENABLE_CONTENT_CHECK:
            if CONTENT_MODE == "exact":
                from deepdiff import DeepDiff
                diff = DeepDiff(d1, d2, ignore_order=True)
                content_ok = (not diff)
                content_detail = {"mode": "exact", "equal": content_ok}
            else:
                flat1 = dict(flatten(d1))
                flat2 = dict(flatten(d2))
                sim = similarity(flat1, flat2)
                content_ok = (sim >= SIMILARITY_THRESHOLD)
                content_detail = {"mode": "threshold", "similarity": round(sim, 4), "threshold": SIMILARITY_THRESHOLD}

        if not content_ok:
            continue

        # record as matched
        matched_paths.add(pathA)
        matched_paths.add(pathB)

        # sample common keys/values (not huge)
        flat1 = dict(flatten(d1))
        flat2 = dict(flatten(d2))
        common_keys = [k for k in flat1.keys() if k in flat2 and flat1[k] == flat2[k]]
        common_sample = [{"path": k, "value": flat1[k]} for k in sorted(common_keys)[:MAX_COMMON_SAMPLED]]

        report_pairs.append({
            "fileA": {"root": labelA, "rel": relA, "name": metaA["name"], "identifier": metaA["identifier"], "tags_count": len(metaA["tags"])},
            "fileB": {"root": labelB, "rel": relB, "name": metaB["name"], "identifier": metaB["identifier"], "tags_count": len(metaB["tags"])},
            "shared_tags": meta_info.get("shared_tags", []),
            "checks": meta_info.get("checks", []),
            "content_check": content_detail,
            "common_values_sampled": common_sample,
        })

    # Copy unique matched files (dedupe by file content hash)
    copied = 0
    seen_hashes = set()
    for abs_p in sorted(matched_paths):
        _, label, _, _, rel = index_by_abs[abs_p]
        try:
            h = file_sha256(abs_p)
        except Exception:
            h = f"path::{abs_p}"
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        dst = build_dest(abs_p, rel, label)
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(abs_p, dst)
        copied += 1

    # Write YAML report
    report = {
        "config": {
            "require_same_name": REQUIRE_SAME_NAME,
            "require_same_identifier": REQUIRE_SAME_IDENTIFIER,
            "tags_match_mode": TAGS_MATCH_MODE,
            "tags_intersect_min": TAGS_INTERSECT_MIN,
            "tags_required": TAGS_REQUIRED,
            "case_insensitive": CASE_INSENSITIVE_COMPARE,
            "field_paths": FIELD_PATHS,
            "match_by_filename": MATCH_BY_FILENAME,
            "preserve_structure": PRESERVE_STRUCTURE,
            "include_intra_root": INCLUDE_INTRA_ROOT,
            "enable_content_check": ENABLE_CONTENT_CHECK,
            "content_mode": CONTENT_MODE if ENABLE_CONTENT_CHECK else "none",
            "similarity_threshold": SIMILARITY_THRESHOLD if ENABLE_CONTENT_CHECK and CONTENT_MODE == "threshold" else None,
            "max_common_sampled": MAX_COMMON_SAMPLED,
        },
        "stats": {
            "input_roots": len(input_dirs),
            "yaml_files_discovered": len(all_files),
            "pairs_compared": total_pairs,
            "unique_files_copied": copied,
        },
        "pairs": report_pairs,
    }
    ensure_dir(matching_common_folder)
    report_path = os.path.join(matching_common_folder, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)

    print("Done.")
    print(f"  Pairs compared        : {total_pairs}")
    print(f"  Unique files copied   : {copied} -> {matching_common_folder}")
    print(f"  Report                : {report_path}")


if __name__ == "__main__":
    main()
