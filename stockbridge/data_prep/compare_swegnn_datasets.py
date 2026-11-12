"""Bit-for-bit comparison of two SWE-GNN dataset folders (reproduction test).

    python stockbridge/data_prep/compare_swegnn_datasets.py \
        --reference .../swegnn/stockbridge_phase2_100_5m_v01 \
        --candidate /tmp/.../phase2_repro \
        --report /tmp/.../reproduction_report.json

Compares the SHA-256 of static_graph.pt and of every events/<split>/<id>.pt
(the file sets must match too). For every file whose bytes differ, it loads
both and reports which keys differ (tensor equality, max |diff|, dtype, shape)
so a failure says *what* differs. manifest.json / manifest.csv are not
compared bit-for-bit (they contain script names and paths); their
event_ids_by_split and cell_size are compared instead.

Exit code 0 only if everything is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files(root: Path) -> list[str]:
    files = [root / "static_graph.pt", *sorted((root / "events").glob("*/*.pt"))]
    return [str(p.relative_to(root)) for p in files if p.is_file()]


def describe_difference(a, b) -> dict:
    """Content-level difference of two loaded objects (dicts of tensors and scalars)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        out = {"dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]}
        if a.shape == b.shape:
            out["equal"] = bool(torch.equal(a, b))
            if not out["equal"] and a.is_floating_point() and b.is_floating_point():
                out["max_abs_diff"] = float((a.double() - b.double()).abs().max())
        else:
            out["equal"] = False
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b), key=str)
        out = {}
        for k in keys:
            if k not in a or k not in b:
                out[str(k)] = {"only_in": "reference" if k in a else "candidate"}
                continue
            d = describe_difference(a[k], b[k])
            if not d.get("equal", True):
                out[str(k)] = d
        return {"equal": not out, "differing_keys": out} if out else {"equal": True}
    equal = (a == b) if type(a) is type(b) else False
    if not isinstance(equal, bool):
        equal = bool(equal)
    return {"equal": equal} if equal else {"equal": False, "reference": repr(a)[:200],
                                           "candidate": repr(b)[:200]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    ref, cand = args.reference.expanduser(), args.candidate.expanduser()

    ref_files, cand_files = relative_files(ref), relative_files(cand)
    report = {
        "reference": str(ref),
        "candidate": str(cand),
        "files_only_in_reference": sorted(set(ref_files) - set(cand_files)),
        "files_only_in_candidate": sorted(set(cand_files) - set(ref_files)),
        "num_reference_files": len(ref_files),
        "num_candidate_files": len(cand_files),
        "identical_files": 0,
        "differing_files": {},
    }
    common = [f for f in ref_files if f in set(cand_files)]
    for i, rel in enumerate(common, 1):
        sha_ref, sha_cand = sha256_file(ref / rel), sha256_file(cand / rel)
        if sha_ref == sha_cand:
            report["identical_files"] += 1
        else:
            content = describe_difference(torch.load(ref / rel, map_location="cpu"),
                                          torch.load(cand / rel, map_location="cpu"))
            report["differing_files"][rel] = {"sha256_reference": sha_ref,
                                              "sha256_candidate": sha_cand,
                                              "content": content}
        if i % 20 == 0 or i == len(common):
            print(f"Compared {i}/{len(common)} files", flush=True)

    manifests = {}
    for name, root in (("reference", ref), ("candidate", cand)):
        path = root / "manifest.json"
        manifests[name] = json.loads(path.read_text()) if path.is_file() else {}
    report["manifest_json"] = {
        key: {"equal": manifests["reference"].get(key) == manifests["candidate"].get(key),
              "reference": manifests["reference"].get(key) if key != "event_ids_by_split" else None,
              "candidate": manifests["candidate"].get(key) if key != "event_ids_by_split" else None}
        for key in ("event_ids_by_split", "cell_size", "rainfall_scale_reference", "split_counts",
                    "num_nodes", "num_edges", "pt_root", "script_name")
    }
    report["reference_manifest_options"] = {k: manifests["reference"].get(k) for k in
                                            ("script_name", "pt_root", "cell_size",
                                             "rainfall_scale_reference", "split_counts")}

    passed = (not report["files_only_in_reference"] and not report["files_only_in_candidate"]
              and not report["differing_files"] and len(common) > 0
              and all(report["manifest_json"][k]["equal"]
                      for k in ("event_ids_by_split", "cell_size", "rainfall_scale_reference")))
    report["bit_identical"] = passed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print("Reference manifest options:", json.dumps(report["reference_manifest_options"]))
    print(f"Identical files: {report['identical_files']}/{len(ref_files)} (reference)")
    for rel, diff in list(report["differing_files"].items())[:10]:
        print("DIFFERS:", rel, json.dumps(diff["content"])[:500])
    for key in ("files_only_in_reference", "files_only_in_candidate"):
        if report[key]:
            print(f"{key}: {report[key][:10]}")
    for key, value in report["manifest_json"].items():
        if not value["equal"]:
            print(f"manifest.json {key} differs: {value['reference']} vs {value['candidate']}")
    print("REPRODUCTION:", "PASS (bit-identical)" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
