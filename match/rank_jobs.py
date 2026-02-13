#!/usr/bin/env python
import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from filtering import filter_job


ROOT = Path(__file__).resolve().parents[1]
RAW_JOBS_FILE = ROOT / "data" / "jobs_raw.jsonl"
RANKED_CSV_FILE = ROOT / "data" / "jobs_ranked.csv"
RANKED_JSON_FILE = ROOT / "data" / "jobs_ranked.json"
RANKED_META_FILE = ROOT / "data" / "jobs_ranked.meta.json"
RESUME_FILE = ROOT / "match" / "resume.txt"
KEYWORDS_FILE = ROOT / "match" / "keywords.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank internship jobs against resume.")
    parser.add_argument("--top", type=int, default=50, help="Number of rows to print in console.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Optional minimum score cutoff.")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_resume(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text or "Paste your plain-text resume content here" in text:
        raise ValueError(f"{path} is empty. Paste your resume text before running rank_jobs.py.")
    return text


def load_keywords(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def recency_score(job: Dict, horizon_days: int = 45) -> float:
    now = dt.datetime.now(dt.timezone.utc)
    timestamps = [job.get("updated_at"), job.get("posted_at")]
    valid: List[dt.datetime] = []
    for ts in timestamps:
        if not ts:
            continue
        try:
            valid.append(parse_dt(ts))
        except ValueError:
            continue
    if not valid:
        return 0.5
    recent = max(valid)
    days = max((now - recent).total_seconds() / 86400.0, 0)
    return max(0.0, min(1.0, 1.0 - (days / horizon_days)))


def text_for_embedding(job: Dict) -> str:
    return "\n".join(
        [
            job.get("title") or "",
            job.get("description_text") or "",
            job.get("location") or "",
        ]
    ).strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize_keyword(kw: str) -> str:
    return kw.lower().strip()


def keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    hits: List[str] = []
    lowered = text.lower()
    for kw in keywords:
        norm = normalize_keyword(kw)
        if not norm:
            continue
        # Treat alphabetic terms as word-boundary patterns for cleaner matching.
        if re.search(r"[a-z]", norm):
            pattern = r"(?<!\w)" + re.escape(norm) + r"(?!\w)"
            if re.search(pattern, lowered):
                hits.append(norm)
        elif norm in lowered:
            hits.append(norm)
    return hits


def compute_keyword_score(job: Dict, cfg: Dict) -> Tuple[float, List[str]]:
    text = f"{job.get('title') or ''} {job.get('description_text') or ''} {job.get('location') or ''}".lower()
    positive = keyword_hits(text, cfg.get("positive_keywords") or [])
    negatives = keyword_hits(text, cfg.get("negative_keywords") or [])
    role_map = cfg.get("role_keywords") or {}

    weighted = 0.0
    matched: Dict[str, float] = {}

    for kw in positive:
        weighted += 1.0
        matched[kw] = max(matched.get(kw, 0.0), 1.0)

    for role_name, words in role_map.items():
        role_hits = keyword_hits(text, words)
        boost = 2.0 if role_name == "ai_ml" else 1.0
        for kw in role_hits:
            weighted += boost
            matched[kw] = max(matched.get(kw, 0.0), boost)

    for kw in negatives:
        weighted -= 2.0
        matched[kw] = max(matched.get(kw, 0.0), 0.5)

    normalized = max(0.0, min(1.0, weighted / 12.0))
    top_keywords = [k for k, _ in sorted(matched.items(), key=lambda item: item[1], reverse=True)[:10]]
    return normalized, top_keywords


def dedupe_jobs(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        key = (row.get("source"), row.get("source_company_token"), row.get("source_job_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def snippet(text: str, limit: int = 260) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:limit]


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "score_total",
        "score_semantic",
        "score_keyword",
        "score_recency",
        "company",
        "title",
        "location",
        "workplace_type",
        "employment_type",
        "season_tag",
        "url",
        "matched_keywords",
        "snippet",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["matched_keywords"] = ", ".join(out.get("matched_keywords") or [])
            writer.writerow(out)


def print_top(rows: List[Dict], top_n: int) -> None:
    if not rows:
        print("No ranked jobs found.")
        return
    print("")
    print(f"Top {min(top_n, len(rows))} jobs")
    print("-" * 110)
    for row in rows[:top_n]:
        print(
            f"#{row['rank']:>3}  {row['score_total']:.4f}  {row['company'][:22]:<22}  "
            f"{row['title'][:48]:<48}  {row['season_tag']:<12}  {row['location'][:24]}"
        )


def main() -> int:
    args = parse_args()

    try:
        resume = load_resume(RESUME_FILE)
        keyword_cfg = load_keywords(KEYWORDS_FILE)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows = read_jsonl(RAW_JOBS_FILE)
    if not rows:
        print(f"No rows found in {RAW_JOBS_FILE}. Run ingest/greenhouse_fetch.py first.")
        return 1
    rows = dedupe_jobs(rows)

    accepted: List[Dict] = []
    rejected_counts: Dict[str, int] = {}
    for row in rows:
        keep, reason, annotations = filter_job(row)
        if not keep:
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        row.update(annotations)
        accepted.append(row)

    if not accepted:
        print("No jobs passed filter criteria.")
        return 0

    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load embedding model '{MODEL_NAME}': {exc}", file=sys.stderr)
        return 1

    resume_vec = model.encode([resume], convert_to_numpy=True, normalize_embeddings=False)[0]
    job_texts = [text_for_embedding(job) for job in accepted]
    job_vecs = model.encode(job_texts, convert_to_numpy=True, batch_size=64, normalize_embeddings=False)

    ranked_rows: List[Dict] = []
    for job, vec in zip(accepted, job_vecs):
        semantic = max(0.0, cosine_similarity(resume_vec, vec))
        keyword_score, matched_keywords = compute_keyword_score(job, keyword_cfg)
        freshness = recency_score(job)
        total = (0.80 * semantic) + (0.15 * keyword_score) + (0.05 * freshness)

        if (job.get("season_tag") or "unknown") == "unknown":
            total *= 0.90
        if not (job.get("location") or "").strip() and (job.get("workplace_type") != "remote"):
            total *= 0.95

        if total < args.min_score:
            continue

        ranked_rows.append(
            {
                "rank": 0,
                "score_total": round(total, 6),
                "score_semantic": round(semantic, 6),
                "score_keyword": round(keyword_score, 6),
                "score_recency": round(freshness, 6),
                "company": job.get("company") or "",
                "title": job.get("title") or "",
                "location": job.get("location") or "",
                "workplace_type": job.get("workplace_type") or "unknown",
                "employment_type": job.get("employment_type") or "unknown",
                "season_tag": job.get("season_tag") or "unknown",
                "url": job.get("url") or "",
                "matched_keywords": matched_keywords,
                "snippet": snippet(job.get("description_text") or ""),
            }
        )

    ranked_rows.sort(key=lambda row: row["score_total"], reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx

    RANKED_JSON_FILE.write_text(json.dumps(ranked_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    write_csv(RANKED_CSV_FILE, ranked_rows)
    RANKED_META_FILE.write_text(
        json.dumps(
            {
                "total_input_jobs": len(rows),
                "accepted_jobs": len(accepted),
                "ranked_jobs": len(ranked_rows),
                "rejected_reasons": rejected_counts,
                "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                "model": MODEL_NAME,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Ranked {len(ranked_rows)} jobs.")
    print(f"Wrote {RANKED_CSV_FILE}")
    print(f"Wrote {RANKED_JSON_FILE}")
    print(f"Wrote {RANKED_META_FILE}")
    print_top(ranked_rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
