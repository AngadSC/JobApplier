# JobApplier MVP: Greenhouse Ingestion + Resume Matching

This repo now has two components:

- `jobsync/`: JobSync UI + tracker (submodule, run separately).
- Custom pipeline in this repo to fetch and rank internship postings for Summer 2026.

## Directory layout

```text
JobApplier/
  jobsync/                    # submodule (tracker UI)
  ingest/
    greenhouse_companies.txt
    custom_urls.txt
    greenhouse_fetch.py
  match/
    filtering.py
    keywords.json
    resume.txt
    rank_jobs.py
  data/
    cache/
    jobs_raw.jsonl
    jobs_ranked.csv
    jobs_ranked.json
  requirements.txt
  README.md
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Input files

- `ingest/greenhouse_companies.txt`: Greenhouse board tokens (one per line).
- `ingest/custom_urls.txt`: Optional manual leads (saved in raw output).
- `match/resume.txt`: Paste the student's plain-text resume content.
- `match/keywords.json`: Keyword boosts/penalties (AI/ML-heavy defaults included).

## Run ingestion

```bash
python ingest/greenhouse_fetch.py
```

Options:

- `--force`: bypass cache and refetch.
- `--max-age-hours 6`: cache age threshold.
- `--token stripe --token databricks`: fetch specific boards only.

Output:

- `data/jobs_raw.jsonl`
- Per-company cache files in `data/cache/greenhouse_<token>.json`

## Run matching + ranking

```bash
python match/rank_jobs.py --top 50
```

Output:

- `data/jobs_ranked.csv`
- `data/jobs_ranked.json`
- `data/jobs_ranked.meta.json` (filter/rejection stats)

## Enforced filtering rules

- Internship-ish only:
  - Include: `intern`, `internship`, `co-op`, `coop`.
  - Exclude if title includes `senior`, `staff`, `principal`, `lead`, `manager`, `director`.
- Time alignment:
  - Accept Summer 2026 or explicit May-Aug 2026 / May-Dec 2026 style text.
  - If ambiguous, keep with `season_tag=unknown` and lower score.
  - Reject explicit non-2026 postings.
- Part-time rule:
  - If part-time, must be remote.
- Geography:
  - Accept Canada/US, remote, or unknown location (unknown is down-ranked).

## Ranking formula

The ranker uses:

- Semantic similarity (`all-MiniLM-L6-v2`) between resume and job text.
- Keyword scoring from `match/keywords.json`.
- Recency score based on `posted_at`/`updated_at`.

Final score:

```text
score_total = 0.80 * semantic + 0.15 * keyword + 0.05 * recency
```

Plus penalties for unknown season/location.

## Workflow with JobSync

1. Run fetch + rank.
2. Review top rows in `data/jobs_ranked.csv`.
3. Manually apply to best-fit jobs.
4. Track application status in `jobsync/` (applied/interview/rejected).

## Safety

- No LinkedIn automation.
- No login automation for private portals.
- Public job sources only (Greenhouse API + manual URLs for now).
