#!/usr/bin/env python
import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
TOKENS_FILE = ROOT / "ingest" / "greenhouse_companies.txt"
CUSTOM_URLS_FILE = ROOT / "ingest" / "custom_urls.txt"
RAW_OUTPUT_FILE = DATA_DIR / "jobs_raw.jsonl"
DEFAULT_CACHE_MAX_AGE_HOURS = 6
GREENHOUSE_URL_TEMPLATE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_list_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    values: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        values.append(cleaned)
    return values


def cache_path_for_token(token: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", token)
    return CACHE_DIR / f"greenhouse_{safe}.json"


def cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = payload.get("fetched_at")
        if not fetched_at:
            return False
        fetched_dt = dt.datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        age = dt.datetime.now(dt.timezone.utc) - fetched_dt
        return age.total_seconds() <= max_age_hours * 3600
    except (ValueError, json.JSONDecodeError):
        return False


def strip_html(raw_html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw_html or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_parse_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()


def detect_workplace_type(location: str, title: str, description: str) -> str:
    merged = f"{location} {title} {description}".lower()
    if "remote" in merged or "work from home" in merged or "wfh" in merged:
        return "remote"
    if "hybrid" in merged:
        return "hybrid"
    if location.strip():
        return "onsite"
    return "unknown"


def detect_employment_type(title: str, description: str) -> str:
    merged = f"{title} {description}".lower()
    if "part-time" in merged or "part time" in merged:
        return "part-time"
    if "intern" in merged or "internship" in merged or "co-op" in merged or "coop" in merged:
        return "internship"
    if "full-time" in merged or "full time" in merged:
        return "full-time"
    return "unknown"


def extract_department(job: Dict) -> Optional[str]:
    departments = job.get("departments") or []
    if departments and isinstance(departments, list):
        first = departments[0]
        if isinstance(first, dict):
            return first.get("name")
    return None


def normalize_greenhouse_job(token: str, job: Dict) -> Dict:
    title = (job.get("title") or "").strip()
    raw_description = job.get("content") or ""
    description_text = strip_html(raw_description)
    location = (job.get("location") or {}).get("name") or ""
    company = (job.get("company_name") or token).strip()
    workplace_type = detect_workplace_type(location, title, description_text)
    employment_type = detect_employment_type(title, description_text)

    source_job_id = str(job.get("id")) if job.get("id") is not None else ""
    url = job.get("absolute_url") or ""
    posted_at = safe_parse_iso(job.get("updated_at") or job.get("first_published"))
    updated_at = safe_parse_iso(job.get("updated_at"))

    return {
        "source": "greenhouse",
        "source_company_token": token,
        "source_job_id": source_job_id,
        "company": company,
        "title": title,
        "location": location,
        "workplace_type": workplace_type,
        "employment_type": employment_type,
        "url": url,
        "posted_at": posted_at,
        "updated_at": updated_at,
        "description_text": description_text,
        "department": extract_department(job),
        "raw": job,
    }


def fetch_greenhouse_jobs(token: str, timeout: int = 30, retries: int = 3) -> Dict:
    last_error = None
    url = GREENHOUSE_URL_TEMPLATE.format(token=token)
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                raise RuntimeError(f"Token '{token}' not found (404).")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed fetching token '{token}': {last_error}") from last_error


def load_or_fetch_token_jobs(token: str, force: bool, max_age_hours: int) -> Dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_token(token)

    if not force and cache_is_fresh(cache_path, max_age_hours):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return {"from_cache": True, "payload": cached.get("payload") or {}}

    payload = fetch_greenhouse_jobs(token)
    cache_payload = {"fetched_at": utc_now_iso(), "payload": payload}
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
    return {"from_cache": False, "payload": payload}


def manual_lead_records(urls: Iterable[str]) -> List[Dict]:
    records: List[Dict] = []
    for idx, url in enumerate(urls, start=1):
        records.append(
            {
                "source": "manual",
                "source_company_token": None,
                "source_job_id": f"manual-{idx}",
                "company": "Unknown",
                "title": "Manual lead",
                "location": "",
                "workplace_type": "unknown",
                "employment_type": "unknown",
                "url": url,
                "posted_at": None,
                "updated_at": None,
                "description_text": "",
                "department": None,
                "raw": {"url": url},
            }
        )
    return records


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True))
            fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch internship postings from Greenhouse boards.")
    parser.add_argument("--force", action="store_true", help="Bypass cache and refetch all tokens.")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=DEFAULT_CACHE_MAX_AGE_HOURS,
        help=f"Cache freshness window (default: {DEFAULT_CACHE_MAX_AGE_HOURS}h).",
    )
    parser.add_argument("--token", action="append", default=[], help="Only fetch this token (repeatable).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokens = args.token or read_list_file(TOKENS_FILE)
    custom_urls = read_list_file(CUSTOM_URLS_FILE)

    if not tokens:
        print(f"No tokens found. Add entries to {TOKENS_FILE}.")
        return 1

    all_jobs: List[Dict] = []
    skipped_tokens: List[str] = []
    used_cache = 0
    fetched = 0

    for token in tokens:
        try:
            result = load_or_fetch_token_jobs(token=token, force=args.force, max_age_hours=args.max_age_hours)
            payload = result["payload"]
            jobs = payload.get("jobs") or []
            for job in jobs:
                all_jobs.append(normalize_greenhouse_job(token, job))
            if result["from_cache"]:
                used_cache += 1
            else:
                fetched += 1
            print(f"[ok] {token}: {len(jobs)} jobs ({'cache' if result['from_cache'] else 'network'})")
        except Exception as exc:  # noqa: BLE001
            skipped_tokens.append(token)
            print(f"[warn] {token}: {exc}", file=sys.stderr)
            continue

    all_jobs.extend(manual_lead_records(custom_urls))
    all_jobs.sort(key=lambda item: ((item.get("source_company_token") or ""), item.get("source_job_id") or ""))
    write_jsonl(RAW_OUTPUT_FILE, all_jobs)

    print("")
    print(f"Wrote {len(all_jobs)} records to {RAW_OUTPUT_FILE}")
    print(f"Fetched from network: {fetched}")
    print(f"Used cache: {used_cache}")
    print(f"Skipped tokens: {len(skipped_tokens)}")
    if skipped_tokens:
        print("Skipped list: " + ", ".join(skipped_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
