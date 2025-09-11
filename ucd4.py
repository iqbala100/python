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
]

# Base destination; global copies go here, per-root copies go under {matching_common_folder}_by_root/<root_label>/
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
ENABLE_CONTENT_CHECK = False      # set True to also check content with DeepDiff
CONTENT_MODE = "exact"            # 'exact' or 'threshold'
SIMILARITY_THRESHOLD = 0.90       # used when CONTENT_MODE == 'threshold'

# Pre-filter: compare only files that share the same filename across roots
MATCH_BY_FILENAME = False

# Copy layout options
PRESERVE_STRUCTURE = False        # mirror source subfolders; if False, flatten with safe names

# Output choices
WRITE_GLOBAL_COMMON = True        # copy all matched files into a single folder (matching_common_folder)
WRITE_PER_ROOT_COMMON = True      # also copy matched files into per-root folders
PER_ROOT_BASE = matching_common_folder + "_by_root"  # base folder for per-root copies

# If True, only copy files whose (name+identifier) are present in *every* input root (intersection across all roots)
REQUIRE_GROUP_IN_ALL_ROOTS = False

# Clear destination folders before copying
CLEAN_DEST = False

# YAML report name (placed in the global folder + per-root base)
REPORT_NAME = "matches_report.yaml"

# Report sample size for common key/values
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
    # dedupe preserve order
    seen, dedup = set(), []
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
    return (len(inter) >= 1, {"shared": inter})

def metadata_match(meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    reasons = []
    if REQUIRE_SAME_NAME:
        if not (meta_a["name"] and meta_b["name"] and meta_a["name"] == meta_b["name"]):
            return (False, {"reason": "name_mismatch", "a": meta_a["name"], "b": meta_b["name"]})
        reasons.append("name_eq")
    if REQUIRE_SAME_IDENTIFIER:
        if not (meta_a["identifier"] and meta_b["identifier"] and meta_a["identifier"] == meta_b["identifier"]):
            return (False, {"reason": "identifier_mismatch", "a": meta_a["identifier"], "b": meta_b["identifier"]})
        reasons.append("identifier_eq")
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

def build_dest_base(global_base: str, rel_p: str, root_label: str, preserve: bool) -> str:
    if preserve:
        dst = os.path.join(global_base, root_label, rel_p)
        ensure_dir(os.path.dirname(dst))
        return dst
    safe = sane_name(rel_p)
    dst = os.path.join(global_base, f"{root_label}__{safe}")
    root, ext = os.path.splitext(dst)
    if not ext:
        dst = root + ".yaml"
    return dst

def write_yaml(path: str, data: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

# ================= main =================
def main():
    if not input_dirs or len(input_dirs) < 2:
        print("Provide at least two input directories in 'input_dirs'.")
        return

    # Prepare destinations
    if WRITE_GLOBAL_COMMON:
        ensure_dir(matching_common_folder)
        if CLEAN_DEST:
            for root, dirs, files in os.walk(matching_common_folder, topdown=False):
                for name in files:
                    try: os.remove(os.path.join(root, name))
                    except Exception: pass
                for name in dirs:
                    try: os.rmdir(os.path.join(root, name))
                    except Exception: pass

    if WRITE_PER_ROOT_COMMON:
        ensure_dir(PER_ROOT_BASE)
        if CLEAN_DEST:
            for root, dirs, files in os.walk(PER_ROOT_BASE, topdown=False):
                for name in files:
                    try: os.remove(os.path.join(root, name))
                    except Exception: pass
                for name in dirs:
                    try: os.rmdir(os.path.join(root, name))
                    except Exception: pass

    # Collect files: (root_idx, label, root, abs, rel)
    all_files: List[Tuple[int, str, str, str, str]] = []
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
                    if a[0] == b[0]:
                        continue  # cross-root only for matching basis
                    pairs.append((a, b))
    else:
        for i in range(len(all_files)):
            for j in range(i + 1, len(all_files)):
                a, b = all_files[i], all_files[j]
                if a[0] == b[0]:
                    continue  # cross-root only by default
                pairs.append((a, b))
    if not pairs:
        print("No cross-root pairs to compare (check MATCH_BY_FILENAME).")
        return

    # Prepare indices/collections
    index_by_abs = {e[3]: e for e in all_files}
    matched_paths_global = set()           # all matched file paths (for global copy)
    matched_paths_by_root: Dict[str, set] = {}  # root_label -> set(abs_paths) for per-root copies
    group_presence: Dict[Tuple[Optional[str], Optional[str]], set] = {}  # (name,identifier) -> set(root_idx)
    report_pairs = []
    total_pairs = 0

    # Compare
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

        ok_meta, meta_info = metadata_match(metaA, metaB)
        if not ok_meta:
            continue

        content_ok = True
        content_detail = {"mode": "none"}
        if ENABLE_CONTENT_CHECK:
            if CONTENT_MODE == "exact":
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

        # Mark presence by (name, identifier)
        key = (metaA["name"], metaA["identifier"])  # metaA == metaB here given rules
        group_presence.setdefault(key, set()).update({idxA, idxB})

        # Record matched paths (global & per-root)
        matched_paths_global.update([pathA, pathB])
        matched_paths_by_root.setdefault(labelA, set()).add(pathA)
        matched_paths_by_root.setdefault(labelB, set()).add(pathB)

        # Sample common key/values (for report)
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

    # If required, restrict to groups present in *all* roots
    if REQUIRE_GROUP_IN_ALL_ROOTS:
        must_have = set(range(len(input_dirs)))
        allowed_groups = {k for k, roots in group_presence.items() if roots.issuperset(must_have)}

        def keep_if_in_all(abs_p: str) -> bool:
            idx, label, root, _, rel = index_by_abs[abs_p]
            try:
                doc = load_first_yaml_doc(abs_p) or {}
            except Exception:
                return False
            meta = extract_metadata(doc)
            return (meta["name"], meta["identifier"]) in allowed_groups

        matched_paths_global = {p for p in matched_paths_global if keep_if_in_all(p)}
        for label in list(matched_paths_by_root.keys()):
            matched_paths_by_root[label] = {p for p in matched_paths_by_root[label] if p in matched_paths_global}

    # ---- Copy: GLOBAL (optional) ----
    copied_global = 0
    if WRITE_GLOBAL_COMMON:
        seen_hashes = set()
        for abs_p in sorted(matched_paths_global):
            idx, label, root, _, rel = index_by_abs[abs_p]
            try:
                h = file_sha256(abs_p)
            except Exception:
                h = f"path::{abs_p}"
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            dst = build_dest_base(matching_common_folder, rel, label, PRESERVE_STRUCTURE)
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(abs_p, dst)
            copied_global += 1

        # global report
        global_report = {
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
                "enable_content_check": ENABLE_CONTENT_CHECK,
                "content_mode": CONTENT_MODE if ENABLE_CONTENT_CHECK else "none",
                "similarity_threshold": SIMILARITY_THRESHOLD if ENABLE_CONTENT_CHECK and CONTENT_MODE == "threshold" else None,
                "require_group_in_all_roots": REQUIRE_GROUP_IN_ALL_ROOTS,
                "max_common_sampled": MAX_COMMON_SAMPLED,
            },
            "stats": {
                "input_roots": len(input_dirs),
                "pairs_compared": total_pairs,
                "unique_files_copied": copied_global,
            },
            "pairs": report_pairs,
        }
        write_yaml(os.path.join(matching_common_folder, REPORT_NAME), global_report)

    # ---- Copy: PER-ROOT (optional) ----
    copied_by_root_summary = {}
    if WRITE_PER_ROOT_COMMON:
        for label, paths in matched_paths_by_root.items():
            per_root_folder = os.path.join(PER_ROOT_BASE, label)
            ensure_dir(per_root_folder)
            seen_hashes = set()
            copied = 0
            for abs_p in sorted(paths):
                # Skip if excluded by all-roots requirement
                if REQUIRE_GROUP_IN_ALL_ROOTS and abs_p not in matched_paths_global:
                    continue
                idx, lbl, root, _, rel = index_by_abs[abs_p]
                try:
                    h = file_sha256(abs_p)
                except Exception:
                    h = f"path::{abs_p}"
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                dst = build_dest_base(per_root_folder, rel, label if not PRESERVE_STRUCTURE else "", PRESERVE_STRUCTURE)
                ensure_dir(os.path.dirname(dst))
                shutil.copy2(abs_p, dst)
                copied += 1
            copied_by_root_summary[label] = copied

        # per-root summary report
        per_root_report = {
            "config_ref": "same-as-global",
            "copied_by_root": copied_by_root_summary,
            "roots": [os.path.basename(os.path.normpath(p)) or f"root{i+1}" for i, p in enumerate(input_dirs)],
        }
        write_yaml(os.path.join(PER_ROOT_BASE, REPORT_NAME), per_root_report)

    print("Done.")
    print(f"  Roots                 : {len(input_dirs)}")
    print(f"  Pairs compared        : {total_pairs}")
    if WRITE_GLOBAL_COMMON:
        print(f"  Global copies         : {copied_global} -> {matching_common_folder}")
    if WRITE_PER_ROOT_COMMON:
        print(f"  Per-root base         : {PER_ROOT_BASE}")
        for lbl, cnt in copied_by_root_summary.items():
            print(f"    {lbl}: {cnt} files")

if __name__ == "__main__":
    main()
