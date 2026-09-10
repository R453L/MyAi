"""
AI News -> X(Twitter) Thread Prep Bot
--------------------------------------
- Fetches AI-related news from multiple free RSS/API sources
- Dedupes against a rolling JSON cache (max 42 entries, FIFO)
- Uses OpenRouter (free model) to generate:
    - a thread with a hook in the first ~10 words
    - an image prompt (only if the story genuinely needs a visual)
- Posts the result + source link to a private Telegram channel via Bot API

All secrets are read from environment variables (set as GitHub Actions secrets).
"""

import os
import json
import time
import random
import hashlib
import requests
import feedparser

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STATE_FILE = "state/posted.json"
MAX_CACHE = 42  # rolling window: oldest link is dropped once this is exceeded

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
)  # change this if the free model gets deprecated/renamed

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/feed/",
    "https://news.google.com/rss/search?q=artificial+intelligence+when:1d&hl=en-US&gl=US&ceid=US:en",
]

HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date?query=AI&tags=story"
ARXIV_API_URL = (
    "http://export.arxiv.org/api/query?search_query=cat:cs.AI"
    "&sortBy=submittedDate&sortOrder=descending&max_results=15"
)

REQUEST_HEADERS = {"User-Agent": "TruelyShocked-AI-News-Bot/1.0"}


# ---------------------------------------------------------------------------
# STATE (dedupe cache)
# ---------------------------------------------------------------------------

def load_state():
    if not os.path.exists(STATE_FILE):
        return []
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # keep only the most recent MAX_CACHE entries (FIFO)
    trimmed = state[-MAX_CACHE:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def link_hash(link: str) -> str:
    return hashlib.sha256(link.strip().lower().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FETCHERS
# ---------------------------------------------------------------------------

def fetch_rss_items():
    items = []
    for url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:15]:
                items.append(
                    {
                        "title": entry.get("title", "").strip(),
                        "link": entry.get("link", "").strip(),
                        "summary": entry.get("summary", "")[:600],
                        "source": url,
                    }
                )
        except Exception as e:
            print(f"[warn] RSS fetch failed for {url}: {e}")
    return items


def fetch_hn_items():
    items = []
    try:
        r = requests.get(HN_ALGOLIA_URL, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        for hit in r.json().get("hits", [])[:20]:
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            items.append(
                {
                    "title": hit.get("title", "").strip(),
                    "link": link,
                    "summary": "",
                    "source": "Hacker News",
                }
            )
    except Exception as e:
        print(f"[warn] HN fetch failed: {e}")
    return items


def fetch_arxiv_items():
    items = []
    try:
        r = requests.get(ARXIV_API_URL, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        parsed = feedparser.parse(r.text)
        for entry in parsed.entries:
            items.append(
                {
                    "title": entry.get("title", "").strip().replace("\n", " "),
                    "link": entry.get("link", "").strip(),
                    "summary": entry.get("summary", "")[:600].replace("\n", " "),
                    "source": "arXiv",
                }
            )
    except Exception as e:
        print(f"[warn] arXiv fetch failed: {e}")
    return items


def collect_all_items():
    items = fetch_rss_items() + fetch_hn_items() + fetch_arxiv_items()
    # basic cleanup: drop empty title/link
    items = [i for i in items if i["title"] and i["link"]]
    random.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# LLM (OpenRouter)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a social media writer for an AI-niche X (Twitter) account.
Audience: people interested in AI tools, AI offers/free trials, AI tutorials, AI news, AI innovation.

Write based ONLY on the following source (do not invent facts not supported by it).
Do NOT copy sentences verbatim from the source — rewrite everything in your own words.

Title: {title}
Summary: {summary}

Return STRICT JSON only, no markdown fences, no extra text, in this exact shape:
{{
  "hook": "first line of the thread, must be attention-grabbing within the first 10 words",
  "thread": ["tweet 1 text (includes the hook)", "tweet 2 text", "tweet 3 text", "... up to 6 tweets total"],
  "needs_image": true or false,
  "image_prompt": "a detailed prompt for an AI image generator if needs_image is true, else empty string"
}}
"""


def generate_thread(item):
    prompt = PROMPT_TEMPLATE.format(title=item["title"], summary=item["summary"] or item["title"])

    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            }
        ),
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # strip accidental code fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_to_telegram(item, thread_data):
    lines = []
    lines.append("🧵 *New AI Thread Draft*")
    lines.append("")
    for i, tweet in enumerate(thread_data["thread"], start=1):
        lines.append(f"{i}. {tweet}")
    lines.append("")
    if thread_data.get("needs_image") and thread_data.get("image_prompt"):
        lines.append("🎨 *Image Prompt (for Nano Banana):*")
        lines.append(thread_data["image_prompt"])
        lines.append("")
    lines.append(f"🔗 *Source (verify before posting):* {item['link']}")

    text = "\n".join(lines)

    r = requests.post(
        url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": False,
        },
        timeout=30,
    )
    r.raise_for_status()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    state = load_state()
    posted_hashes = {entry["hash"] for entry in state}

    candidates = collect_all_items()
    print(f"[info] fetched {len(candidates)} candidate items")

    for item in candidates:
        h = link_hash(item["link"])
        if h in posted_hashes:
            continue

        try:
            thread_data = generate_thread(item)
        except Exception as e:
            print(f"[warn] LLM generation failed for '{item['title']}': {e}")
            continue

        try:
            send_to_telegram(item, thread_data)
        except Exception as e:
            print(f"[error] Telegram send failed: {e}")
            continue

        state.append({"hash": h, "link": item["link"], "title": item["title"], "ts": int(time.time())})
        save_state(state)
        print(f"[ok] posted: {item['title']}")
        return  # one fresh item per run (6 runs/day = 6 posts/day)

    print("[info] no fresh un-posted item found this run")


if __name__ == "__main__":
    main()
