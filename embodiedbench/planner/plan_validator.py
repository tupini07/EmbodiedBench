"""Plan validation and repair utilities for EmbodiedBench.

Updated precedence (name-first):
    We now PREFER the textual `action_name` semantics over the numeric `action_id`
    when both are present but disagree. Rationale: empirical logs show the model
    frequently emits a *sensible* action_name paired with an incorrect ID. The
    previous logic treated the ID as source of truth and overwrote the name; this
    lost semantic intent. New policy:

        1. Attempt to resolve the action_name to a canonical form (exact match,
             case/spacing-insensitive, or synonym). If resolvable, repair the ID to
             match the canonical name (and canonicalize the name if needed).
        2. Only if the name cannot be resolved do we fall back to using a canonical
             ID (if valid) to repair the name.
        3. Otherwise mark the step unrecoverable.

Environment variable controls:
    PLAN_REPAIR_MODE:
        - "off"    : Do nothing (return original string)
        - "log"    : Validate, log report, do NOT modify output
        - "apply"  : Validate, repair inconsistencies, return repaired JSON
                                    (default if unset)
    PLAN_REPAIR_ADD_META:
        - If set to "1", inject a `_repairs_meta` field listing repairs.

Repairs performed:
    * If action_name canonical (or via synonym / normalized) but id mismatched -> fix id.
    * Synonyms resolved via normalization (lowercase, strip spaces/underscores).
    * If name not resolvable but id canonical -> fix name from id.
    * Case / spacing canonicalization of name recorded as a repair.
    * Leaves unrecoverable steps untouched and records them.

Defensive guarantees:
    * Never adds NEW steps.
    * Never reorders steps.
    * Only mutates fields within existing step dicts.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from .action_catalog import ACTION_ID_TO_NAME, ACTION_NAME_TO_ID, SYNONYMS

NORMALIZE_REMOVE = {" ", "_"}

# Build a normalized lookup (lowercase + remove spaces/underscores) for all canonical names.
# This supports case-insensitive + spacing-insensitive direct resolution without needing
# every variant in SYNONYMS. Does NOT include suffixed numbered variants separately, since
# they already appear explicitly in ACTION_NAME_TO_ID.
ACTION_NORMALIZED_NAME_MAP = { }
for _n in ACTION_NAME_TO_ID.keys():  # canonical names as-is
    ACTION_NORMALIZED_NAME_MAP.setdefault("".join(ch for ch in _n.lower() if ch not in NORMALIZE_REMOVE), _n)


def _normalize(s: str) -> str:
    s2 = s.lower()
    for ch in NORMALIZE_REMOVE:
        s2 = s2.replace(ch, "")
    return s2


def _canonical_from_synonym(raw: str) -> str | None:
    key = _normalize(raw)
    return SYNONYMS.get(key)


def validate_and_repair_plan(raw_json: str) -> Tuple[str, Dict[str, Any]]:
    """Validate / repair a model JSON output containing an `executable_plan`.

    Returns (possibly_modified_json_string, report_dict).
    The report contains:
      parse_ok: bool
      actions_seen: int
      repairs: list
      unrecoverable: list
      error: optional parse error
    """
    report: Dict[str, Any] = {
        "parse_ok": False,
        "actions_seen": 0,
        "repairs": [],
        "unrecoverable": [],
    }
    # Strip non-standard comments outside of string literals to avoid parse failures.
    # Supported patterns now:
    #   // line comments
    #   /* block comments */
    #   #  python style line comments
    #   <!-- html style block comments -->
    def _strip_comments(s: str) -> str:
        out = []
        i = 0
        in_string = False
        escape = False
        while i < len(s):
            ch = s[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                out.append(ch)
                i += 1
                continue
            # --- Line comments (// ... ) ---
            if ch == '/' and i + 1 < len(s) and s[i+1] == '/':
                i += 2
                while i < len(s) and s[i] not in '\n\r':
                    i += 1
                continue
            # --- Block comments (/* ... */) ---
            if ch == '/' and i + 1 < len(s) and s[i+1] == '*':
                i += 2
                while i + 1 < len(s) and not (s[i] == '*' and s[i+1] == '/'):
                    i += 1
                i += 2 if i + 1 < len(s) else 1
                continue
            # --- Python style (# ...) ---
            if ch == '#':
                # Treat everything until newline as comment.
                i += 1
                while i < len(s) and s[i] not in '\n\r':
                    i += 1
                continue
            # --- HTML style (<!-- ... -->) ---
            if ch == '<' and i + 3 < len(s) and s[i+1:i+4] == '!--':
                i += 4
                # Scan until closing '-->'
                while i + 2 < len(s) and not (s[i] == '-' and s[i+1] == '-' and s[i+2] == '>'):
                    i += 1
                i += 3 if i + 2 < len(s) else (len(s) - i)
                continue
            out.append(ch)
            i += 1
        return ''.join(out)

    sanitized_json = _strip_comments(raw_json)
    try:
        obj = json.loads(sanitized_json)
    except Exception as e:  # JSON parse failure—cannot proceed.
        report["error"] = f"json_parse_failed: {e}"
        return raw_json, report

    report["parse_ok"] = True
    plan = obj.get("executable_plan")
    if not isinstance(plan, list):
        return raw_json, report

    repaired = False
    for idx, step in enumerate(plan):
        if not isinstance(step, dict):
            report["unrecoverable"].append({"index": idx, "reason": "non_dict"})
            continue
        if "action_id" not in step or "action_name" not in step:
            report["unrecoverable"].append({"index": idx, "reason": "missing_keys"})
            continue
        aid = step["action_id"]
        aname = step["action_name"]
        report["actions_seen"] += 1

        if not isinstance(aname, str):
            report["unrecoverable"].append({"index": idx, "reason": "name_not_str"})
            continue

        # --- NAME-FIRST RESOLUTION ---
        canonical_name: str | None = None

        # (a) Exact canonical match
        if aname in ACTION_NAME_TO_ID:
            canonical_name = aname
        else:
            # (b) Synonym mapping
            syn = _canonical_from_synonym(aname)
            if syn and syn in ACTION_NAME_TO_ID:
                canonical_name = syn
            else:
                # (c) Case/spacing-insensitive mapping
                norm_key = "".join(ch for ch in aname.lower() if ch not in NORMALIZE_REMOVE)
                if norm_key in ACTION_NORMALIZED_NAME_MAP:
                    canonical_name = ACTION_NORMALIZED_NAME_MAP[norm_key]

        if canonical_name is not None:
            correct_id = ACTION_NAME_TO_ID[canonical_name]
            # Repair name canonicalization first (if differs)
            if aname != canonical_name:
                print(f"[plan_validator][repair] step {idx}: canonicalize name '{aname}' -> '{canonical_name}' (id stays {aid})")
                step["action_name"] = canonical_name
                repaired = True
                report["repairs"].append({
                    "index": idx,
                    "type": "canonicalize_name",
                    "old_name": aname,
                    "new_name": canonical_name,
                })
            # Repair ID if mismatched
            if aid != correct_id:
                print(f"[plan_validator][repair] step {idx}: id_from_name mismatch id {aid} -> {correct_id} for name '{canonical_name}'")
                step["action_id"] = correct_id
                repaired = True
                report["repairs"].append({
                    "index": idx,
                    "type": "id_from_name",
                    "old_id": aid,
                    "new_id": correct_id,
                })
            continue

        # Fallback: name not resolvable, but ID may be canonical.
        if isinstance(aid, int) and aid in ACTION_ID_TO_NAME:
            canonical_from_id = ACTION_ID_TO_NAME[aid]
            if canonical_from_id != aname:
                print(f"[plan_validator][repair] step {idx}: name_from_id fallback '{aname}' -> '{canonical_from_id}' (id {aid})")
                step["action_name"] = canonical_from_id
                repaired = True
                report["repairs"].append({
                    "index": idx,
                    "type": "name_from_id_fallback",
                    "old_name": aname,
                    "new_name": canonical_from_id,
                })
            continue

        # Unable to repair either field.
        print(f"[plan_validator][unrecoverable] step {idx}: could not resolve pair action_id={aid} action_name='{aname}'")
        report["unrecoverable"].append({
            "index": idx,
            "reason": "unrecognized_pair",
            "action_id": aid,
            "action_name": aname,
        })

    if not repaired:
        return raw_json, report

    if os.getenv("PLAN_REPAIR_ADD_META", "0") == "1":
        # Non-destructive meta only if not already present.
        if "_repairs_meta" not in obj:
            obj["_repairs_meta"] = report["repairs"]
    return json.dumps(obj, ensure_ascii=False), report


def maybe_repair(raw_json: str) -> str:
    """Optionally repair a plan JSON string depending on mode and environment.

    Environment gating:
        Only apply repairs for Habitat and Alfred style environments where
        the textual action_name <-> numeric action_id catalog exists and
        semantic correction is meaningful. For navigation (eb-nav) and
        manipulation (eb-man) environments, action schemas differ (e.g.,
        pure id lists or continuous vectors) so we skip all repair logic.

        The active environment is read from EB_ENV_NAME (set in main.py).
        Whitelist: {"eb-hab", "eb-alf"}
    """
    env_name = os.getenv("EB_ENV_NAME", "")
    if env_name not in {"eb-alf", "eb-hab"}:
        # Bypass validation entirely for unsupported environments.
        return raw_json
    
    mode = os.getenv("PLAN_REPAIR_MODE", "apply").lower()
    if mode not in {"off", "log", "apply"}:
        mode = "apply"

    if mode == "off":
        return raw_json

    repaired_json, rep = validate_and_repair_plan(raw_json)
    if env_name and os.getenv("PLAN_REPAIR_MODE", "apply") != "off":
        print(f"[plan_validator] env={env_name} mode={mode} parse_ok={rep.get('parse_ok')} actions_seen={rep.get('actions_seen')} repairs={len(rep.get('repairs', []))} unrecoverable={len(rep.get('unrecoverable', []))}")
        if rep.get("repairs"):
            for r in rep["repairs"]:
                print(f"  [plan_validator][repairs] index={r['index']} type={r['type']} detail={ {k:v for k,v in r.items() if k not in ['index','type']} }")
        if rep.get("unrecoverable"):
            for u in rep["unrecoverable"]:
                print(f"  [plan_validator][unrecoverable] index={u['index']} reason={u['reason']} id={u.get('action_id')} name='{u.get('action_name')}'")
    # Always log a concise summary when there is any mismatch.
    if rep.get("repairs") or rep.get("unrecoverable"):
        logging.debug(
            "[plan_validator] mode=%s parse_ok=%s actions=%s repairs=%s unrecoverable=%s",
            mode,
            rep.get("parse_ok"),
            rep.get("actions_seen"),
            len(rep.get("repairs", [])),
            len(rep.get("unrecoverable", [])),
        )
    if mode == "apply":
        return repaired_json
    return raw_json

__all__ = ["maybe_repair", "validate_and_repair_plan"]
