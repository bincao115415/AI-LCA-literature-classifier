"""
AI-LCA literature classification pipeline using DeepSeek chat completion API.

Workflow:
1) Read a Web of Science Excel export (columns: Title, Abstract, Author Keywords, Keywords Plus).
2) For each record, ask an LLM to (a) decide if the paper is AI-LCA per the rules, (b) assign one
   or more Groups and Subgroups (separate columns), with rationale.
3) Write results to a new Excel/CSV with structured fields (is_ai_lca, groups[], subgroups[], rationale, confidence, llm_raw).

Usage:
  python ai_lca_classifier.py --input wos.xlsx --output wos_classified.xlsx --api-key YOUR_KEY
or set environment variable DEEPSEEK_API_KEY instead of --api-key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

from urllib3.exceptions import NotOpenSSLWarning
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import pandas as pd
import requests
from requests import Response
from requests.adapters import HTTPAdapter


SYSTEM_PROMPT = """
You are an expert in AI-for-LCA. Read the full title/abstract/keywords and reason by SEMANTICS, not just keyword matches. Follow these strict rules:

Step 0: A paper is AI-LCA only if BOTH are true:
 (A) It uses AI/ML/deep learning/large language model/optimization algorithms; AND
 (B) These AI methods are substantively integrated into at least one step of the LCA workflow
     (LCI handling, characterization factor/emission factor prediction, impact prediction,
      optimization, uncertainty analysis, dynamic integration).
If AI is unrelated to LCA, or LCA is only background/one of many KPIs for a non-LCA task, mark not_ai_lca.
If the study runs standard LCA with no AI inside LCA workflow, mark not_ai_lca.
If you cannot find explicit evidence of environmental indicators or LCA workflow integration, default to is_ai_lca=false and explain briefly.
If "life cycle" only refers to product lifetime, mechanical fatigue/reliability, or usage cycles without environmental LCA indicators, mark not_ai_lca.

If AI-LCA, list ALL applicable Groups and Subgroups from:
 Group 1 – Goal, Scope & Inventory: 1.1 Literature Mining & Scoping; 1.2 LCI Data Extraction,
  Updating & Imputation; 1.3 Data Quality Assessment & Reconciliation (tag, primary still 1.2).
 Group 2 – Characterization & Emission Factor Prediction: 2.1 Molecular Toxicity & Environmental
  Fate Prediction; 2.2 Emission Factors & Other Characterization Factors.
 Group 3 – Streamlined / Surrogate LCA: 3.1 Product/Process-level Surrogate LCA; 3.2 Spatial /
  Scenario Surrogates & High-throughput Screening.
 Group 4 – Optimization, Uncertainty & Interpretation: 4.1 Multi-objective Optimization &
  Eco-design; 4.2 Uncertainty Quantification & Global Sensitivity; 4.3 Explainable & Causal AI for
  LCA Interpretation.
Group 5 – Time-explicit / Dynamic Integration: 5.1 Dynamic / Time-explicit LCA Modelling;
 5.2 Digital Twin & Operational Decision Support.

Assignment preference:
- Always include every applicable Group/Subgroup. No primary/secondary fields; just lists.
- If AI builds surrogate/approximation models (SSM/DSM/DNN surrogate/meta-model/proxy) to predict LCA indicators,
  ALWAYS include Group 3 (3.1 or 3.2). If optimization/interpretation is also present, also include Group 4 subgroups.
- If AI surrogate/optimizer directly predicts or optimizes environmental indicators (e.g., carbon footprint) alongside
  cost/technical metrics, it IS AI-LCA and must include Group 3.

Important exclusions:
- Techno-economic analysis (TEA) or process/intensity metrics without environmental LCIA indicators -> not AI-LCA.
- "Life cycle" meaning mechanical/fatigue lifetime, stress/strain/displacement, or reliability-only is NOT environmental LCA.
- Digital twin that only predicts mechanical or operational states without computing environmental indicators is NOT AI-LCA.

Return STRICT JSON ONLY (no prose). Schema:
{
  "is_ai_lca": true|false,
  "groups": ["Group X", ...],
  "subgroups": ["X.Y ...", ...],
  "rationale": "Concise evidence-based reasoning (2-4 sentences)",
  "confidence": 0.0-1.0
}

Use semantics, not keyword matching. Be concise and deterministic. Temperature=0.
"""


USER_PROMPT_TEMPLATE = """Title: {title}
Abstract: {abstract}
Author Keywords: {author_kw}
Keywords Plus: {keywords_plus}"""


API_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class LLMConfig:
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 5
    sleep_between_retries: float = 1.5
    backoff_multiplier: float = 2.0
    pool_connections: int = 16
    pool_maxsize: int = 32
    status_forcelist: Sequence[int] = (429, 500, 502, 503, 504)


def build_session(cfg: LLMConfig) -> requests.Session:
    """Create a shared session with connection pooling and basic HTTP retries."""
    session = requests.Session()
    try:
        retry = Retry(
            total=cfg.max_retries,
            read=cfg.max_retries,
            connect=cfg.max_retries,
            backoff_factor=0.5,
            status_forcelist=cfg.status_forcelist,
            allowed_methods=frozenset({"POST"}),
        )
    except TypeError:
        # Fallback for older urllib3 that uses method_whitelist
        retry = Retry(
            total=cfg.max_retries,
            read=cfg.max_retries,
            connect=cfg.max_retries,
            backoff_factor=0.5,
            status_forcelist=cfg.status_forcelist,
            method_whitelist=frozenset({"POST"}),
        )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=cfg.pool_connections,
        pool_maxsize=cfg.pool_maxsize,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def call_deepseek(messages: List[Dict[str, str]], cfg: LLMConfig, session: requests.Session) -> str:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        expand_tokens = False
        try:
            resp: Response = session.post(API_URL, headers=headers, json=payload, timeout=cfg.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            if not content.strip():
                expand_tokens = finish_reason == "length" or bool(message.get("reasoning_content"))
                reasoning_len = len(message.get("reasoning_content") or "")
                raise RuntimeError(
                    f"Empty content from model (finish_reason={finish_reason}, reasoning_tokens={reasoning_len})"
                )
            return content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < cfg.max_retries:
                if expand_tokens:
                    payload["max_tokens"] = min(int(payload["max_tokens"] * 1.5), 4096)
                sleep_time = cfg.sleep_between_retries * (cfg.backoff_multiplier ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"DeepSeek API failed after {cfg.max_retries} attempts") from exc
    raise last_error  # type: ignore[misc]


def safe_parse_json(content: str) -> Dict[str, Any]:
    """Parse model output; try to recover JSON if wrapped."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to find the first {...} block.
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            return json.loads(snippet)
        raise


def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def normalize_confidence(value: Any) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0:
        return 0.0
    if conf > 1:
        return 1.0
    return conf


def clean_text(value: Any) -> str:
    """Normalize missing values to empty strings and strip whitespace."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def build_user_prompt(row: pd.Series) -> str:
    title = clean_text(row.get("Title") or row.get("Article Title"))
    return USER_PROMPT_TEMPLATE.format(
        title=title,
        abstract=clean_text(row.get("Abstract")),
        author_kw=clean_text(row.get("Author Keywords")),
        keywords_plus=clean_text(row.get("Keywords Plus")),
    )


def classify_record(row: pd.Series, cfg: LLMConfig, session: requests.Session) -> Dict[str, Any]:
    text_blob = " ".join(
        [
            clean_text(row.get("Title") or row.get("Article Title")),
            clean_text(row.get("Abstract")),
            clean_text(row.get("Author Keywords")),
            clean_text(row.get("Keywords Plus")),
        ]
    ).lower()

    user_prompt = build_user_prompt(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = call_deepseek(messages, cfg, session)

    # Debug: Print raw response for the first few records to debug issues
    import random
    if random.random() < 0.1:  # Print ~10% of responses
        print(f"\n[DEBUG] Raw LLM response for '{row.get('Title', '')[:30]}...':\n{raw[:500]}...\n")

    parsed = safe_parse_json(raw)
    groups: List[str] = normalize_list(parsed.get("groups"))
    subgroups: List[str] = normalize_list(parsed.get("subgroups"))
    confidence = normalize_confidence(parsed.get("confidence"))

    # Surrogate keyword post-check: ensure Group 3 is present when surrogate hints appear.
    surrogate_hints = ["surrogate", "proxy", "meta-model", "metamodel", "approximation", "ssm", "dsm"]
    has_surrogate = any(h in text_blob for h in surrogate_hints)
    if parsed.get("is_ai_lca") and has_surrogate and all("Group 3" not in g for g in groups):
        groups.append("Group 3")
        subgroups.append("3.1 Product/Process-level Surrogate LCA")

    is_ai_lca = normalize_bool(parsed.get("is_ai_lca"))
    rationale = parsed.get("rationale")

    # Confidence thresholding for manual review
    if confidence is not None and confidence < 0.6:
        is_ai_lca = None
        note = "low confidence, needs manual review"
        rationale = f"{rationale}; {note}" if rationale else note

    return {
        "is_ai_lca": is_ai_lca,
        "groups": dedupe_preserve(groups),
        "subgroups": dedupe_preserve(subgroups),
        "rationale": rationale,
        "confidence": confidence,
        "llm_raw": raw,
    }


def load_api_key(cli_key: Optional[str]) -> str:
    # Priority: CLI arg > Env var
    key = cli_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("ERROR: Provide --api-key or set DEEPSEEK_API_KEY")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-LCA literature classifier (DeepSeek).")
    parser.add_argument("--input", required=True, help="Input Web of Science Excel file")
    parser.add_argument("--output", required=True, help="Output Excel/CSV path")
    parser.add_argument("--api-key", help="DeepSeek API key (or set DEEPSEEK_API_KEY)")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows for testing")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent workers for API calls")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Checkpoint frequency (rows)")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per record")
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Backoff multiplier between retries (sleep * multiplier^(attempt-1))",
    )
    parser.add_argument(
        "--only-failures-from",
        help="Path to previous output; reprocess rows where rationale contains 'Classification failed' or is missing.",
    )
    return parser.parse_args()


def failure_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "rationale" in df.columns:
        mask = mask | df["rationale"].astype(str).str.contains("Classification failed", na=False)
    if "is_ai_lca" in df.columns:
        mask = mask | df["is_ai_lca"].isna()
    return mask


def warn_missing_columns(df: pd.DataFrame) -> None:
    expected_cols = {"Title", "Article Title", "Abstract", "Author Keywords", "Keywords Plus"}
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(
            f"Warning: missing columns {', '.join(missing)}; classifier will treat them as empty strings.",
            file=sys.stderr,
        )


def write_llm_jsonl(results: List[Dict[str, Any]], path: str, orig_indices: Optional[Sequence[int]] = None) -> None:
    """Persist raw LLM outputs for audit."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            for i, res in enumerate(results):
                if not res:
                    continue
                raw = res.get("llm_raw")
                if raw is None:
                    continue
                payload = {
                    "row_index": int(orig_indices[i]) if orig_indices is not None else i,
                    "is_ai_lca": res.get("is_ai_lca"),
                    "groups": res.get("groups"),
                    "subgroups": res.get("subgroups"),
                    "confidence": res.get("confidence"),
                    "llm_raw": raw,
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to write LLM raw log to {path}: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    try:
        api_key = load_api_key(args.api_key)
    except ValueError as exc:
        sys.exit(str(exc))
    cfg = LLMConfig(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_between_retries=1.5,
        backoff_multiplier=args.retry_backoff,
    )
    session = build_session(cfg)

    base_df: Optional[pd.DataFrame] = None
    RESULT_COLS = ["is_ai_lca", "groups", "subgroups", "rationale", "confidence", "llm_raw"]

    if args.only_failures_from:
        base_df = pd.read_excel(args.only_failures_from)
        mask = failure_mask(base_df)
        df = base_df.loc[mask].copy()
        df["__orig_idx"] = df.index
        if len(df) == 0:
            print(f"No failed rows found in {args.only_failures_from}; nothing to do.")
            return
        print(f"Retrying {len(df)} failed rows from {args.only_failures_from}...")
    else:
        df = pd.read_excel(args.input)
        if args.max_rows:
            df = df.head(args.max_rows)

    warn_missing_columns(df)

    results: List[Dict[str, Any]] = [None] * len(df)
    checkpoint_interval = args.checkpoint_every
    max_workers = max(1, min(args.max_workers, len(df)))

    print(f"Processing {len(df)} records with {max_workers} concurrent workers...")
    print(f"Checkpointing every {checkpoint_interval} completed records...")

    import concurrent.futures
    from tqdm import tqdm

    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(classify_record, row, cfg, session): i 
            for i, (_, row) in enumerate(df.iterrows())
        }
        
        # Use tqdm for progress bar
        with tqdm(total=len(df), unit="rec", desc="Classifying") as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "is_ai_lca": None,
                        "groups": None,
                        "subgroups": None,
                        "rationale": f"Classification failed: {exc}",
                        "confidence": None,
                        "llm_raw": str(exc),
                    }
                
                results[i] = result
                completed_count += 1
                pbar.update(1)

                # Checkpoint logic
                if completed_count % checkpoint_interval == 0:
                    pbar.set_description(f"Classifying (Saved chkpt @ {completed_count})")
                    # ... checkpoint saving code ...

                temp_res_df = pd.DataFrame([r if r is not None else {} for r in results])

                if base_df is not None:
                    temp_full = base_df.copy()
                    # Update only rows being retried
                    idx_map = df["__orig_idx"]
                    for res, orig_idx in zip(results, idx_map):
                        if res is None:
                            continue
                        for col in RESULT_COLS:
                            temp_full.at[orig_idx, col] = res.get(col)
                    temp_out = temp_full
                else:
                    temp_out = pd.concat([df.reset_index(drop=True), temp_res_df], axis=1)

                if args.output.lower().endswith(".csv"):
                    temp_path = f"{args.output}.checkpoint.csv"
                    temp_out.to_csv(temp_path, index=False)
                else:
                    temp_path = f"{args.output}.checkpoint.xlsx"
                    temp_out.to_excel(temp_path, index=False)
                    
                    # Restore clean description
                    pbar.set_description("Classifying")
    
    print("\nProcessing complete. Saving final output...")

    res_df = pd.DataFrame(results)
    if base_df is not None:
        # Merge back into previous results
        out_df = base_df.copy()
        idx_map = df["__orig_idx"]
        for res, orig_idx in zip(results, idx_map):
            if res is None:
                continue
            for col in RESULT_COLS:
                out_df.at[orig_idx, col] = res.get(col)
        out_df = out_df.drop(columns=["__orig_idx"], errors="ignore")
    else:
        out_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

    if args.output.lower().endswith(".csv"):
        out_df.to_csv(args.output, index=False)
    else:
        out_df.to_excel(args.output, index=False)

    # Persist raw LLM responses for audit
    log_path = f"{args.output}.llm_raw.jsonl"
    try:
        idx_map_for_log = df.get("__orig_idx") if "__orig_idx" in df.columns else None
    except Exception:
        idx_map_for_log = None
    write_llm_jsonl(results, log_path, idx_map_for_log)

    session.close()


if __name__ == "__main__":
    main()
