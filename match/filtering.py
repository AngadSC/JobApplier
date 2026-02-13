import re
from typing import Dict, Tuple


INTERN_INCLUDE_RE = re.compile(r"\b(intern|internship|co-?op)\b", re.IGNORECASE)
TITLE_EXCLUDE_RE = re.compile(r"\b(senior|staff|principal|lead|manager|director)\b", re.IGNORECASE)
PART_TIME_RE = re.compile(r"\b(part[- ]?time)\b", re.IGNORECASE)
REMOTE_RE = re.compile(r"\b(remote|work from home|wfh)\b", re.IGNORECASE)
SEASON_SUMMER_2026_RE = re.compile(r"\bsummer\s*2026\b", re.IGNORECASE)
SEASON_2026_INTERNSHIP_RE = re.compile(r"\b2026\b.{0,25}\b(intern|internship|co-?op)\b", re.IGNORECASE)
MONTH_2026_RE = re.compile(r"\b(may|june|july|aug(?:ust)?)\b.{0,20}\b2026\b", re.IGNORECASE)
EIGHT_MONTH_RE = re.compile(r"\b(8|eight)\s*[- ]?\s*month\b", re.IGNORECASE)
BAD_SEASON_RE = re.compile(r"\bsummer\s*(20[0-9]{2})\b", re.IGNORECASE)
BAD_INTERNSHIP_YEAR_RE = re.compile(
    r"\b(20[0-9]{2})\b.{0,24}\b(intern|internship|co-?op)\b|\b(intern|internship|co-?op)\b.{0,24}\b(20[0-9]{2})\b",
    re.IGNORECASE,
)
BAD_MONTH_YEAR_RE = re.compile(r"\b(may|june|july|aug(?:ust)?)\b.{0,20}\b(20[0-9]{2})\b", re.IGNORECASE)

US_HINTS = {
    "usa",
    "united states",
    "u.s.",
    "us",
}

CANADA_HINTS = {
    "canada",
    "ontario",
    "quebec",
    "british columbia",
    "alberta",
    "manitoba",
    "saskatchewan",
    "nova scotia",
    "new brunswick",
    "newfoundland",
    "pei",
    "prince edward island",
}

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

CA_PROVINCE_CODES = {"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"}


def detect_season_tag(text: str) -> str:
    lowered = text.lower()
    if SEASON_SUMMER_2026_RE.search(lowered) or SEASON_2026_INTERNSHIP_RE.search(lowered):
        return "summer_2026"
    if MONTH_2026_RE.search(lowered):
        if EIGHT_MONTH_RE.search(lowered):
            return "may-dec_2026"
        return "summer_2026"
    return "unknown"


def is_timeframe_reject(text: str) -> bool:
    # Reject only when season/internship timing explicitly points to a non-2026 cycle.
    for match in BAD_SEASON_RE.findall(text):
        if int(match) != 2026:
            return True

    for match in BAD_MONTH_YEAR_RE.findall(text):
        year = int(match[1])
        if year != 2026:
            return True

    for match in BAD_INTERNSHIP_YEAR_RE.findall(text):
        # The year appears in either group 1 or group 4.
        year_text = match[0] or match[3]
        if year_text and int(year_text) != 2026:
            return True
    return False


def is_internshipish(title: str, description: str) -> bool:
    combined = f"{title} {description}"
    if not INTERN_INCLUDE_RE.search(combined):
        return False
    if TITLE_EXCLUDE_RE.search(title):
        return False
    return True


def is_remote(location: str, workplace_type: str, text: str) -> bool:
    if workplace_type == "remote":
        return True
    merged = f"{location} {text}"
    return bool(REMOTE_RE.search(merged))


def is_part_time(text: str) -> bool:
    return bool(PART_TIME_RE.search(text))


def has_region_hint(location: str) -> bool:
    lowered = location.lower()
    if not lowered.strip():
        return False
    if any(hint in lowered for hint in US_HINTS | CANADA_HINTS):
        return True

    tokenized = re.findall(r"[A-Z]{2}", location.upper())
    for tok in tokenized:
        if tok in US_STATE_CODES or tok in CA_PROVINCE_CODES:
            return True
    return False


def location_allowed(location: str, workplace_type: str) -> bool:
    if workplace_type == "remote":
        return True
    if not location.strip():
        return True
    return has_region_hint(location)


def filter_job(job: Dict) -> Tuple[bool, str, Dict]:
    title = job.get("title") or ""
    desc = job.get("description_text") or ""
    location = job.get("location") or ""
    workplace_type = job.get("workplace_type") or "unknown"
    full_text = f"{title} {desc} {location}"

    season_tag = detect_season_tag(full_text)
    annotations = {"season_tag": season_tag}

    if not is_internshipish(title, desc):
        return False, "not_internshipish", annotations
    if is_timeframe_reject(full_text):
        return False, "wrong_year", annotations
    if is_part_time(full_text) and not is_remote(location, workplace_type, full_text):
        return False, "part_time_not_remote", annotations
    if not location_allowed(location, workplace_type):
        return False, "outside_geography", annotations
    return True, "accepted", annotations
