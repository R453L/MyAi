"""
Weekly AI Tools/Trends Roundup
------------------------------
Runs once a week (separate cron from the main 4-hourly script).

Unlike a hand-written "top 20 tools" list, this pulls LIVE rankings each time
it runs:
  - Hacker News: AI-related stories from the last 7 days, sorted by points
  - Product Hunt: today's AI category ranking (the feed itself is already
    ranked by daily popularity)

Nothing here is a fixed list of tool/account names — the ranking sources
decide what's trending, so this keeps working the same way whether it's
2026 or 2126: whatever is popular THAT WEEK is what gets pulled in.

Reuses helpers (state cache, Telegram send, HTML escaping, OpenRouter creds)
from fetch_and_post.py, which must live in the same directory.
"""

import time
import json
import requests

from fetch_and_post import (
    load_state,
    save_state,
    link_hash,
    _tg_send,
    _escape_html,
    call_llm,
    REQUEST_HEADERS,
)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
PRODUCTHUNT_FEED = "https://www.producthunt.com/feed?category=ai"
SEVEN_DAYS_SECONDS = 7 * 24 * 60 * 60
MAX_ITEMS = 5


def fetch_hn_weekly_top(min_points=60, limit=8):
    """Live points-based ranking — whatever the community upvoted THIS week."""
    since = int(time.time()) - SEVEN_DAYS_SECONDS
    params = {
        "query": "AI",
        "tags": "story",
        "numericFilters": f"created_at_i>{since},points>{min_points}",
        "hitsPerPage": 50,
    }
    try:
        r = requests.get(HN_SEARCH_URL, params=params, headers=REQUEST_HEADERS, timeout=15)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        hits.sort(key=lambda h: h.get("points", 0), reverse=True)
        items = []
        for hit in hits[:limit]:
            link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            items.append(
                {
                    "title": hit.get("title", "").strip(),
                    "link": link,
                    "source": "Hacker News (community ranked)",
                    "points": hit.get("points", 0),
                }
            )
        return items
    except Exception as e:
        print(f"[warn] HN weekly fetch failed: {e}")
        return []


def fetch_producthunt_top(limit=8):
    """The feed is already ranked by today's Product Hunt popularity."""
    import feedparser

    items = []
    try:
        parsed = feedparser.parse(PRODUCTHUNT_FEED)
        for entry in parsed.entries[:limit]:
            items.append(
                {
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", "").strip(),
                    "source": "Product Hunt (today's ranking)",
                    "points": 0,
                }
            )
    except Exception as e:
        print(f"[warn] Product Hunt fetch failed: {e}")
    return items


def pick_fresh_items(candidates, posted_hashes, limit=MAX_ITEMS):
    fresh = []
    for c in candidates:
        if not c["title"] or not c["link"]:
            continue
        h = link_hash(c["link"])
        if h in posted_hashes:
            continue
        c["_hash"] = h
        fresh.append(c)
        if len(fresh) >= limit:
            break
    return fresh


ROUNDUP_PROMPT = """You are writing a weekly AI roundup thread for an X (Twitter) account in the AI niche.

Below are {n} AI tools/stories that are trending THIS WEEK based on live rankings (Hacker News
community points, Product Hunt daily rank). This list is different every week — do not treat any
of these as permanent facts about the tool, only describe what's given.

Items:
{items_block}

Write an original roundup thread:
- Post 1: a hook that frames this as this week's trending AI finds. First 10 words must hook.
- Then ONE post per item: say what it is and why it's worth a look, based ONLY on the title/context
  given. Do NOT invent specific features, pricing, or capabilities not implied by the title — if the
  title is vague, keep that post short and honest instead of making things up.
- Final post: a question or hot take to the audience. No filler ("stay tuned", etc).
- Each post under 270 characters.
- Never copy the titles verbatim — rewrite them naturally in your own words.
- No clichéd AI phrases ("game-changer", "unlock the power of", "in today's fast-paced world").
  Vary sentence rhythm. Have a point of view, don't just list facts.

Return STRICT JSON only, no markdown fences:
{{
  "posts": ["post 1 (intro/hook)", "post 2 (item 1)", "... one per item ...", "final post"]
}}
"""


def generate_roundup(items):
    items_block = "\n".join(
        f"{i+1}. {it['title']} (via {it['source']})" for i, it in enumerate(items)
    )
    prompt = ROUNDUP_PROMPT.format(n=len(items), items_block=items_block)
    return call_llm(prompt, temperature=0.85)


def send_roundup_to_telegram(items, roundup_data):
    posts = roundup_data["posts"]
    header = f"📅 <b>Weekly AI Roundup Draft</b> ({len(posts)} posts)"
    _tg_send(header)

    for i, post in enumerate(posts, start=1):
        label = f"Post {i}/{len(posts)}"
        _tg_send(f"{label}\n<pre>{_escape_html(post)}</pre>")

    links_lines = [f"🔗 <b>Sources (verify each before posting):</b>"]
    for it in items:
        links_lines.append(f"• {_escape_html(it['title'])}: {it['link']}")
    _tg_send("\n".join(links_lines), disable_preview=True)


def main():
    state = load_state()
    posted_hashes = {entry["hash"] for entry in state}

    candidates = fetch_hn_weekly_top() + fetch_producthunt_top()
    print(f"[info] fetched {len(candidates)} weekly candidates")

    items = pick_fresh_items(candidates, posted_hashes, limit=MAX_ITEMS)
    if len(items) < 3:
        print("[info] not enough fresh trending items this week, skipping roundup")
        return

    try:
        roundup_data = generate_roundup(items)
    except Exception as e:
        print(f"[error] LLM roundup generation failed: {e}")
        return

    try:
        send_roundup_to_telegram(items, roundup_data)
    except Exception as e:
        print(f"[error] Telegram send failed: {e}")
        return

    for it in items:
        state.append({"hash": it["_hash"], "link": it["link"], "title": it["title"], "ts": int(time.time())})
    save_state(state)
    print(f"[ok] posted weekly roundup with {len(items)} items")


if __name__ == "__main__":
    main()
