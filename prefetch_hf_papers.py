"""Pre-fetch HuggingFace daily papers and cache locally.

Fetches from https://huggingface.co/api/daily_papers?date=YYYY-MM-DD
starting from yesterday, going backwards (weekdays only) until:
  - hitting the lower bound (2026-01-01), or
  - receiving HTTP 429 (rate limited).

Recent 7 days are always refreshed (upvotes may change).
Older dates are skipped if already cached.

Cache file: hf_cache/daily_papers.json
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta

import requests

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)

HF_DAILY_API = 'https://huggingface.co/api/daily_papers'
CACHE_DIR = 'hf_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'daily_papers.json')
LOWER_BOUND = '2025-01-01'
REFRESH_DAYS = 30
REQUEST_DELAY = 1.0


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_daily_papers(date_str: str):
    """Fetch papers for a given date. Returns None on 429."""
    url = f'{HF_DAILY_API}?date={date_str}&limit=100'
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            logging.warning('HTTP 429 on %s — stopping.', date_str)
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            return None
        logging.error('HTTP error for %s: %s', date_str, e)
        return []
    except Exception as e:
        logging.error('Error fetching %s: %s', date_str, e)
        return []


def parse_paper(entry: dict) -> dict:
    """Extract relevant fields from a daily_papers API entry."""
    p = entry.get('paper', {})
    return {
        'id': p.get('id', ''),
        'title': p.get('title', ''),
        'authors': [a.get('name', '') for a in p.get('authors', [])],
        'upvotes': p.get('upvotes', 0),
        'github_repo': p.get('githubRepo') or None,
        'github_stars': p.get('githubStars') or None,
        'published_at': p.get('publishedAt', ''),
        'num_comments': entry.get('numComments', 0),
    }


def is_weekday(d: datetime) -> bool:
    return d.weekday() < 5  # Mon=0 .. Fri=4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lower-bound',
        default=LOWER_BOUND,
        help='Oldest date to fetch (default: %(default)s)',
    )
    parser.add_argument(
        '--refresh-days',
        type=int,
        default=REFRESH_DAYS,
        help='Always refresh this many recent days (default: %(default)s)',
    )
    args = parser.parse_args()

    cache = load_cache()
    lower = datetime.strptime(args.lower_bound, '%Y-%m-%d')
    today = datetime.now()
    current = today - timedelta(days=1)  # start from yesterday
    refresh_cutoff = today - timedelta(days=args.refresh_days)

    fetched = 0
    skipped = 0

    while current >= lower:
        if not is_weekday(current):
            current -= timedelta(days=1)
            continue

        date_str = current.strftime('%Y-%m-%d')
        needs_refresh = current >= refresh_cutoff

        if date_str in cache and not needs_refresh:
            skipped += 1
            current -= timedelta(days=1)
            continue

        time.sleep(REQUEST_DELAY)
        raw = fetch_daily_papers(date_str)

        if raw is None:  # 429
            logging.info(
                'Rate limited. Fetched %d dates, skipped %d cached.',
                fetched,
                skipped,
            )
            save_cache(cache)
            return

        papers = {p['id']: p for p in (parse_paper(e) for e in raw) if p['id']}
        cache[date_str] = {
            'fetched_at': datetime.now().isoformat(),
            'count': len(papers),
            'papers': papers,
        }
        fetched += 1
        logging.info(
            'Fetched %s: %d papers (total fetched: %d)',
            date_str,
            len(papers),
            fetched,
        )

        current -= timedelta(days=1)

    save_cache(cache)
    logging.info(
        'Done. Fetched %d dates, skipped %d cached. Cache has %d dates total.',
        fetched,
        skipped,
        len(cache),
    )


if __name__ == '__main__':
    main()
