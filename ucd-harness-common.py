#!/usr/bin/env python3
"""
Common YAMLs by (name + identifier) AND identical content.

All outputs live ONLY in:
  - <OUTPUT_BASE>\services\shared_common_services\
      - copied YAMLs (filenames are just the original base name)
      - common_index.yaml
      - common_report.yaml
  - <OUTPUT_BASE>\pipelines\shared_common_pipelines\
      - copied YAMLs (filenames are just the original base name)
      - common_index.yaml
      - common_report.yaml
"""

import os
import re
import shutil
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from deepdiff import DeepHash, DeepDiff

# ---------------- CONFIG ----------------

PIPELINES_DIRS = [
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG1\.harness\pipelines",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG2\.harness\pipelines",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG3\.harness\pipelines",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG4\.harness\pipelines",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG5\.harness\pipelines",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG6\.harness\pipelines",
]

SERVICES_DIRS = [
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG1\.harness\services",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG2\.harness\services",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG3\.harness\services",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG4\.harness\services",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG5\.harness\services",
    r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\harness_out_RG6\.harness\services",
]

OUTPUT_BASE = r"C:\Users\hiiqb\Desktop\ucd-compare\ucd-to-harness1\common_output_shared"
CLEAN_DEST = False

CASE_INSENSITIVE_COMPARE = True
REQUIRE_BOTH_FIELDS = True

# Enforce identical content after (name, identifier) match
REQUIRE_IDENTICAL_CONTENT = True
CONTENT_COMPARE_MODE = "hash"        # "hash" or "deepdiff"
IGNORE_ORDER_IN_CONTENT = True       # ignore list/dict order in content compare

FIELD_PATHS = {
    "services": {
        "name":       ["service.name", "name", "metadata.name"],
        "identifier": ["service.identifier", "identifier", "metadata.identifier", "id"],
    },
    "pipelines": {
        "name":       ["pipeline.name", "name", "metadata.name"],
        "identifier": ["pipeline.identifier", "identifier", "metadata.identifier", "id"],
    },
    "_generic": {
        "name":       ["name", "metadata.name"],
        "identifier": ["identifier", "metadata.identifier", "id"],
    }
}

SHARED_SUBDIR = {
    "services":  "shared_common_services",
    "pipelines": "shared_common_pipelines",
}

# ---------------- HELPERS ----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_folder(path: str):
    if not os.path.isdir(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try: os.remove(os.path.join(root, name))
            except Exception: pass
        for name in dirs:
            try: os.rmdir(os.path.join(root, name))
            except Exception: pass

def list_yaml_files_recursively(root_dir: str) -> List[Tuple[str, str]]:
    out = []
    for r, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith((".yaml", ".yml")):
                ab = os.path.join(r, name)
                rel = os.path.relpath(ab, root_dir)
                out.append((ab, rel))
    return out

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

def extract_name_identifier(doc: Any, category: str) -> Tuple[Optional[str], Optional[str]]:
    fp = FIELD_PATHS.get(category, {})
    name_paths = fp.get("name", []) + FIELD_PATHS["_generic"]["name"]
    id_paths   = fp.get("identifier", []) + FIELD_PATHS["_generic"]["identifier"]
    name = norm_str(find_first(doc, name_paths))
    ident = norm_str(find_first(doc, id_paths))
    return name, ident

def deep_hash_signature(obj: Any) -> str:
    try:
        h = DeepHash(obj, ignore_order=IGNORE_ORDER_IN_CONTENT)
        return str(h[obj])
    except Exception:
        dumped = yaml.safe_dump(obj, sort_keys=True, allow_unicode=True)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def content_equal(a: Any, b: Any) -> bool:
    if CONTENT_COMPARE_MODE == "hash":
        return deep_hash_signature(a) == deep_hash_signature(b)
    diff = DeepDiff(a, b, ignore_order=IGNORE_ORDER_IN_CONTENT)
    return not diff

def _trim_leading_harness(name: str) -> str:
    """Remove leading '.harness__' if present (safety net)."""
    return name[len(".harness__"):] if name.startswith(".harness__") else name

def _ensure_yaml_ext(fname: str) -> str:
    if not fname.lower().endswith((".yaml", ".yml")):
        return fname + ".yaml"
    return fname

def _unique_target_path(base_dir: str, fname: str) -> str:
    """
    Return a unique path in base_dir. If file exists with different content,
    append ' (2)', ' (3)', ... before extension.
    """
    root, ext = os.path.splitext(fname)
    candidate = os.path.join(base_dir, fname)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{root} ({n}){ext}")
        n += 1
    return candidate

def _sha256_bytes(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def copy_keep_filename(abs_path: str, rel_from_root: str, out_dir: str, seen_hashes: set) -> Optional[str]:
    """
    Copy to out_dir keeping only the BASE FILE NAME.
    - Dedup identical content across roots by sha256 (skip duplicates).
    - If a different file with the same name already exists, create a unique '(2)', '(3)' filename.
    - Trim any '.harness__' prefix just in case (shouldn't normally appear on real source names).
    Returns destination path or None if skipped due to dedupe.
    """
    ensure_dir(out_dir)
    base_name = os.path.basename(rel_from_root)
    base_name = _trim_leading_harness(base_name)
    base_name = _ensure_yaml_ext(base_name)

    # dedupe by content: if this blob hash already copied, skip
    try:
        src_hash = _sha256_bytes(abs_path)
    except Exception:
        src_hash = f"path::{abs_path}"
    if src_hash in seen_hashes:
        return None

    # if target name exists, avoid overwriting with different content
    dst = os.path.join(out_dir, base_name)
    if os.path.exists(dst):
        try:
            if _sha256_bytes(dst) == src_hash:
                seen_hashes.add(src_hash)
                return dst  # already present with same content
        except Exception:
            pass
        dst = _unique_target_path(out_dir, base_name)

    shutil.copy2(abs_path, dst)
    seen_hashes.add(src_hash)
    return dst

def write_yaml(path: str, data: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

# ---------------- CORE ----------------

def process_category(category: str, roots: List[str], out_base: str) -> Dict[str, Any]:
    """
    - Scan all roots (>=2) for YAMLs in this category.
    - Extract (name, identifier) and group.
    - Require identical content within each (name, identifier) group cluster.
    - Copy ONLY to the shared folder, keeping just the base filename.
    - Write common_index.yaml and common_report.yaml ONLY in the shared folder.
    """
    assert len(roots) >= 2, f"Need at least two input roots for {category}"

    shared_dir = os.path.join(out_base, category, SHARED_SUBDIR[category])
    ensure_dir(shared_dir)
    if CLEAN_DEST:
        clean_folder(shared_dir)

    labels = [Path(r).parent.name or "root" for r in roots]  # environment labels, used only in report

    entries: List[Dict[str, Any]] = []
    skipped_missing_meta = 0

    # Scan
    for idx, root in enumerate(roots):
        for ab, rel in list_yaml_files_recursively(root):
            try:
                doc = load_first_yaml_doc(ab) or {}
            except Exception:
                continue
            name, ident = extract_name_identifier(doc, category)
            if REQUIRE_BOTH_FIELDS and (not name or not ident):
                skipped_missing_meta += 1
                continue
            entries.append({
                "root_idx": idx,
                "root_label": labels[idx],
                "root": root,
                "abs": ab,
                "rel": rel,
                "name": name,
                "identifier": ident,
                "doc": doc,
                "src_base": os.path.basename(rel),
            })

    # Group by (name, identifier)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        groups[(e["name"], e["identifier"])].append(e)

    root_count = len(roots)
    detailed_any: List[Dict[str, Any]] = []
    detailed_shared: List[Dict[str, Any]] = []

    # Track copied outputs
    seen_hashes: set = set()
    copies_log: List[Dict[str, Any]] = []  # for including saved filename in index

    # Build clusters per group by content signature
    for (name, ident), items in groups.items():
        cluster_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for it in items:
            sig_for_id = deep_hash_signature(it["doc"])
            cluster_map[sig_for_id].append(it)

        for cluster_id, cluster in cluster_map.items():
            # Strict verify if deepdiff requested
            if REQUIRE_IDENTICAL_CONTENT and CONTENT_COMPARE_MODE == "deepdiff":
                base_doc = cluster[0]["doc"]
                if any(not content_equal(base_doc, it["doc"]) for it in cluster[1:]):
                    continue

            roots_in_cluster = sorted({it["root_idx"] for it in cluster})
            roots_labels = sorted({labels[i] for i in roots_in_cluster})

            if len(roots_in_cluster) >= 2:
                files_meta = []
                for it in sorted(cluster, key=lambda x: (x["root_idx"], x["rel"])):
                    dst_path = copy_keep_filename(it["abs"], it["rel"], shared_dir, seen_hashes)
                    saved_name = os.path.basename(dst_path) if dst_path else None
                    files_meta.append({
                        "root_label": it["root_label"],
                        "root_index": it["root_idx"],
                        "src_rel": it["rel"],
                        "src_base": it["src_base"],
                        "saved_filename": saved_name
                    })
                    if saved_name:
                        copies_log.append({
                            "saved_filename": saved_name,
                            "from_root": it["root_label"],
                            "src_rel": it["rel"]
                        })

                rec = {
                    "name": name,
                    "identifier": ident,
                    "cluster_id": cluster_id,
                    "roots_present": roots_labels,
                    "files": files_meta
                }
                detailed_any.append(rec)

                if len(roots_in_cluster) == root_count:
                    detailed_shared.append(rec)

    # Write per-category index + report ONLY in shared_dir
    index_payload = {
        "category": category,
        "shared_folder": shared_dir,
        "roots": [{"index": i, "label": labels[i], "path": roots[i]} for i in range(root_count)],
        "matching_basis": "Common when YAML share the same (name, identifier) AND identical content.",
        "config": {
            "case_insensitive": CASE_INSENSITIVE_COMPARE,
            "require_both_fields": REQUIRE_BOTH_FIELDS,
            "require_identical_content": REQUIRE_IDENTICAL_CONTENT,
            "content_compare_mode": CONTENT_COMPARE_MODE,
            "ignore_order_in_content": IGNORE_ORDER_IN_CONTENT,
            "field_paths": FIELD_PATHS[category] if category in FIELD_PATHS else FIELD_PATHS["_generic"],
            "file_naming": "base filename only; collisions get ' (2)', ' (3)' suffixes",
        },
        "counts": {
            "total_yaml_seen": len(entries) + skipped_missing_meta,
            "usable_yaml": len(entries),
            "skipped_missing_name_or_identifier": skipped_missing_meta,
            "distinct_name_identifier_pairs": len(groups),
            "groups_common_any": len(detailed_any),
            "groups_shared_common": len(detailed_shared),
            "files_copied": len({c['saved_filename'] for c in copies_log if c['saved_filename']}),
        },
        "groups_common_any": detailed_any,          # >=2 roots
        "groups_shared_common": detailed_shared,    # == all roots
        "copied_files": copies_log                  # includes saved filenames
    }
    write_yaml(os.path.join(shared_dir, "common_index.yaml"), index_payload)

    report_payload = {
        "category": category,
        "shared_folder": shared_dir,
        "summary": index_payload["counts"],
        "notes": [
            "Copied filenames keep ONLY the original base name.",
            "If two different files would have the same name, a numeric suffix is added.",
            "See common_index.yaml for exact saved filenames and sources."
        ]
    }
    write_yaml(os.path.join(shared_dir, "common_report.yaml"), report_payload)

    # Return minimal stats for console
    return {
        "category": category,
        "shared_dir": shared_dir,
        "files_copied": index_payload["counts"]["files_copied"],
        "groups_any": len(detailed_any),
        "groups_all": len(detailed_shared),
    }

# ---------------- MAIN ----------------

def main():
    ensure_dir(OUTPUT_BASE)

    services_stats  = process_category("services",  SERVICES_DIRS,  OUTPUT_BASE)
    pipelines_stats = process_category("pipelines", PIPELINES_DIRS, OUTPUT_BASE)

    print("Done.")
    print(f"  Services  -> {services_stats['shared_dir']}")
    print(f"    Files copied : {services_stats['files_copied']}")
    print(f"    Groups (any) : {services_stats['groups_any']}")
    print(f"    Groups (all) : {services_stats['groups_all']}")
    print(f"  Pipelines -> {pipelines_stats['shared_dir']}")
    print(f"    Files copied : {pipelines_stats['files_copied']}")
    print(f"    Groups (any) : {pipelines_stats['groups_any']}")
    print(f"    Groups (all) : {pipelines_stats['groups_all']}")

if __name__ == "__main__":
    main()
