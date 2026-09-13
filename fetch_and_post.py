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
import datetime
import requests
import feedparser

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STATE_FILE = "state/posted.json"
MAX_CACHE = 42  # rolling window: oldest link is dropped once this is exceeded

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
# Optional: force a specific model first (rarely needed now — auto-picked below).
# Leave this GitHub Secret empty/unset to fully auto-select the best working free model.
OPENROUTER_MODEL_OVERRIDE = os.environ.get("OPENROUTER_MODEL", "").strip()

# Used only if OpenRouter's live model list can't be fetched at all (network hiccup).
STATIC_FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

RSS_SOURCES = [
    # (url, category_hint) — category_hint is just a suggestion; the LLM makes the final call
    ("https://techcrunch.com/category/artificial-intelligence/feed/", "news"),
    ("https://venturebeat.com/category/ai/feed/", "news"),
    ("https://www.technologyreview.com/feed/", "news"),
    ("https://news.google.com/rss/search?q=artificial+intelligence+when:1d&hl=en-US&gl=US&ceid=US:en", "news"),
    ("https://openai.com/news/rss.xml", "offer"),          # product/feature/tier announcements
    ("https://www.producthunt.com/feed?category=ai", "tool"),
    ("https://huggingface.co/blog/feed.xml", "tutorial"),
    ("https://deepmind.google/blog/feed/basic/", "innovation"),
    ("https://rss.arxiv.org/rss/cs.AI", "innovation"),
    ("http://machinelearningmastery.com/blog/feed", "tutorial"),
]

HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date?query=AI&tags=story"

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
    for url, category_hint in RSS_SOURCES:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:15]:
                items.append(
                    {
                        "title": entry.get("title", "").strip(),
                        "link": entry.get("link", "").strip(),
                        "summary": entry.get("summary", "")[:600],
                        "source": url,
                        "category_hint": category_hint,
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
                    "category_hint": "mixed",
                }
            )
    except Exception as e:
        print(f"[warn] HN fetch failed: {e}")
    return items


def collect_all_items():
    items = fetch_rss_items() + fetch_hn_items()
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
Do NOT mirror the source's exact list order, wording, or structure if it's a list — reorganize and
rephrase so the output is a genuinely original piece of writing, not a reformatted copy.

AVOID SOUNDING LIKE AI:
- No clichéd AI phrases ("in today's fast-paced world", "unlock the power of", "game-changer",
  "let's dive in", "at the end of the day", "it's important to note").
- Vary sentence length and rhythm — don't make every sentence the same structure.
- Avoid overusing em dashes and semicolons. Write the way a sharp, opinionated human would type on X.
- Have a point of view. Don't just describe — react to it.

Title: {title}
Summary: {summary}
Suggested category (you may override if the content clearly fits a different one): {category_hint}

FIRST, pick the real category from: news, tool, tutorial, offer, innovation.
- news: general AI development/event, no specific product angle
- tool: a specific product/model/library people can go try
- tutorial: something actionable — a technique, workflow, or how-to
- offer: an explicit free tier, free trial, discount, or waitlist mentioned in the source
- innovation: a research result or technical breakthrough and why it matters
Only use "offer" if the source text ACTUALLY mentions a free trial/tier/discount/waitlist — never invent one.

THEN decide the format:
- Consider a thread seriously whenever the story has more than one worthwhile angle — a feature +
  a limitation, a claim + the context around it, a launch + what it replaces. Don't default to single
  just because it's easier; only use single when there is genuinely just one point to make.
{format_directive}

STYLE BY CATEGORY:
- tool: lead with what it does + one standout feature; end implying it's worth trying (no fake claims,
  no "link in bio" — the source link goes in a separate message).
- tutorial: frame as a direct, actionable tip the reader can use right now.
- offer: lead with the specific deal/free-tier detail; create mild urgency without being spammy.
- news / innovation: lead with the single most surprising or consequential fact.

HARD RULES:
- Post 1 (or the single post) must have a hook in the first 10 words that makes someone stop scrolling.
  Use a specific number, surprising fact, or bold claim from the source — not a vague generic opener.
- If it's a thread, EVERY post must add NEW information or a NEW angle. Never restate or rephrase a
  point already made in an earlier post. If you can't come up with a genuinely new angle, cut the thread
  short instead of padding it.
- Do NOT end with filler like "stay tuned", "more insights soon", "what a time to be alive". End with
  either: a sharp question to the audience, a contrarian/hot take, or one concrete practical takeaway.
- Each post must be under 270 characters (X limit is 280, leave buffer).
- Use 1-2 relevant emojis per post where they add punch (not every sentence, not decorative spam,
  not the same emoji every time — pick ones that fit the specific content).
- Include 1-3 relevant hashtags (e.g. #AI, #AITools, #ChatGPT) in the LAST post of the single post or
  thread. Hashtags render as clickable blue links on X and help visibility even though they don't
  drive the algorithm much — so include them even though they're not essential to reach.
- Plain, direct, confident tone. No corporate hedging language ("it is fascinating to see", "highlights").
- IMAGE DECISION: Set needs_image true whenever a visual would plausibly boost engagement — this
  includes most tool/product posts (show the tool's interface/concept), most tutorial posts (show the
  technique/result), and news/innovation posts with a concrete visual concept (a robot, a chart-like
  idea, a before/after). Only set needs_image false when the content is genuinely abstract with
  nothing visual to depict (e.g. a pure policy debate, an abstract opinion take with no concrete
  imagery). When in doubt, default to true — err toward including an image prompt, not skipping it.
- IMAGE STYLE: this account posts news/tutorials/tool reviews, not fiction — so image prompts should
  describe a clean PRESENTATION/INFOGRAPHIC/DASHBOARD style, not a hyper-realistic cinematic render.
  Explicitly include what text/labels should appear in the image (a short title, 2-4 short
  labels/callouts relevant to THIS specific story) since modern AI image tools (Gemini/Nano Banana,
  ChatGPT/GPT Image, etc.) render text well now — a wordless "artistic" image is a missed opportunity.
- IMAGE VARIETY — IMPORTANT: do NOT default to the same dark-navy-blue tech-dashboard look every time.
  Randomly pick a genuinely different visual direction for THIS image from options like: (a) light
  background, clean minimal editorial infographic with a bold accent color, (b) warm color palette
  (orange/cream/terracotta) flat-design infographic, (c) colorful flat vector illustration style with
  icons, (d) high-contrast monochrome (black/white/one accent color) poster style, (e) soft pastel
  UI-mockup style. Pick whichever fits the content, but vary it — a bot that always posts the same
  navy-blue dashboard look becomes obviously recognizable as automated, which hurts the account.

Return STRICT JSON only, no markdown fences, no extra text, in this exact shape:
{{
  "category": "news" or "tool" or "tutorial" or "offer" or "innovation",
  "format": "single" or "thread",
  "posts": ["post 1 text (includes the hook)", "post 2 text", "... up to 6 total, 1 if format is single"],
  "needs_image": true or false,
  "image_prompt": "a detailed prompt for an AI image generator if needs_image is true, else empty string"
}}
"""


def generate_thread(item, format_directive=""):
    prompt = PROMPT_TEMPLATE.format(
        title=item["title"],
        summary=item["summary"] or item["title"],
        category_hint=item.get("category_hint", "news"),
        format_directive=format_directive,
    )
    return call_llm(prompt, temperature=0.8)


MULTI_ITEM_PROMPT = """You are writing an X (Twitter) post/thread for an AI-niche account. Category: {category}.

Below are {n} recent {category} items pulled from live sources just now. Decide how many are genuinely
worth covering together — could be just 1 if only one is strong, or several as a listicle-style thread
if multiple are each worth a short mention. Don't force in a weak/vague item just to hit a count.

Items:
{items_block}

Write original content:
- If covering multiple items: come up with a bold, catchy TITLE for this post/thread (its own field,
  not inside the posts). Post 1 is a hook intro. Then one post per item you kept — say what it is and
  why it matters, based ONLY on the title/context given, don't invent specifics. Final post ends with
  a question or hot take + 1-3 hashtags.
- If only 1 item is genuinely worth it: leave title empty, write normal single/thread content with the
  same hook standards as usual.
- Never copy titles verbatim — rewrite naturally. No clichéd AI phrases ("game-changer", "unlock the
  power of"). Vary sentence rhythm. Have a point of view, don't just list facts.
- 1-2 relevant emojis per post (not spammy). Each post under 270 characters.
- IMAGE: set needs_image true when a visual would help (default true for tool/multi-item content) in a
  clean presentation/infographic style with short on-image text labels — and VARY the visual style and
  color palette from whatever you'd typically default to (don't always use a dark navy tech-dashboard
  look; mix in light/warm/vector/monochrome styles across posts so the account doesn't look templated).

Return STRICT JSON only, no markdown fences:
{{
  "title": "bold title if multi-item, else empty string",
  "format": "single" or "thread",
  "posts": ["post 1", "post 2", "..."],
  "dropped_titles": ["exact title text of any item you decided to drop, if any"],
  "needs_image": true or false,
  "image_prompt": "a detailed prompt for an AI image generator if needs_image is true, else empty string"
}}
"""


def generate_multi_item_post(items, category):
    items_block = "\n".join(f"{i+1}. {it['title']}" for i, it in enumerate(items))
    prompt = MULTI_ITEM_PROMPT.format(category=category, n=len(items), items_block=items_block)
    return call_llm(prompt, temperature=0.85)


def get_candidate_models(limit=6):
    """
    Live free-model list from OpenRouter, newest/highest-capacity first.
    This is fetched fresh every run, so it self-adjusts as OpenRouter's free
    lineup changes — no manual updates needed when a free model gets retired.
    """
    candidates = []
    if OPENROUTER_MODEL_OVERRIDE:
        candidates.append(OPENROUTER_MODEL_OVERRIDE)

    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models", headers=REQUEST_HEADERS, timeout=15
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        free = [m for m in data if str(m.get("id", "")).endswith(":free")]
        # simple capability heuristic: bigger context window first
        free.sort(key=lambda m: m.get("context_length", 0) or 0, reverse=True)
        for m in free[:limit]:
            mid = m["id"]
            if mid not in candidates:
                candidates.append(mid)
    except Exception as e:
        print(f"[warn] could not fetch live OpenRouter model list: {e}")

    for mid in STATIC_FALLBACK_MODELS:
        if mid not in candidates:
            candidates.append(mid)

    return candidates


def call_llm(prompt, temperature=0.8):
    """
    Tries candidate free models in order until one returns valid JSON.
    Handles the "this free model got shut down/renamed" problem automatically —
    no manual Secret updates needed when OpenRouter rotates its free lineup.
    """
    last_err = None
    for model_id in get_candidate_models():
        try:
            resp = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    }
                ),
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            parsed = json.loads(raw)
            print(f"[info] used model: {model_id}")
            return parsed
        except Exception as e:
            last_err = e
            print(f"[warn] model '{model_id}' failed, trying next candidate: {e}")
            continue

    raise RuntimeError(f"All candidate models failed. Last error: {last_err}")


def _is_true(value) -> bool:
    """Robust boolean check — some models return "true"/"false" as strings,
    which Python would otherwise treat as truthy regardless of content."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tg_send(text: str, disable_preview: bool = True, _retries: int = 3):
    """
    Sends one Telegram message, with a small pacing delay and automatic
    retry-after handling for flood control (429). Threads/multi-item posts
    send several messages back-to-back — without this, a burst can hit
    Telegram's flood limit partway through and silently drop the rest,
    even though the GitHub Actions run itself still reports success.
    """
    for attempt in range(_retries):
        r = requests.post(
            url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": disable_preview,
            },
            timeout=30,
        )
        if r.status_code == 429:
            retry_after = 3
            try:
                retry_after = r.json().get("parameters", {}).get("retry_after", 3)
            except Exception:
                pass
            print(f"[warn] Telegram flood control hit, waiting {retry_after}s and retrying")
            time.sleep(retry_after + 1)
            continue
        r.raise_for_status()
        time.sleep(0.6)  # small pacing gap so bursts of messages don't trip flood control
        return

    raise RuntimeError("Telegram send failed after retries (flood control)")


def send_to_telegram(item, thread_data):
    """
    Sends a header message, then ONE message per post, each wrapped in an
    HTML <pre> block. In Telegram, tapping a <pre>/code block shows a copy
    icon that copies just that block's text — so on mobile it's a single
    tap to grab a clean post ready to paste into X, no manual selecting.
    """
    posts = thread_data["posts"]
    is_thread = thread_data.get("format") == "thread" and len(posts) > 1
    category = thread_data.get("category", "news").upper()

    kind_label = "Thread" if is_thread else "Single Post"
    header = f"🧵 <b>New AI {kind_label} Draft</b> · #{category} ({len(posts)} post{'s' if len(posts) > 1 else ''})"
    _tg_send(header)

    for i, post in enumerate(posts, start=1):
        label = f"Post {i}/{len(posts)}" if is_thread else "Tap to copy:"
        block = f"{label}\n<pre>{_escape_html(post)}</pre>"
        _tg_send(block)

    if _is_true(thread_data.get("needs_image")) and thread_data.get("image_prompt"):
        img_msg = f"🎨 <b>Image Prompt (use with ChatGPT/Gemini/Nano Banana):</b>\n<pre>{_escape_html(thread_data['image_prompt'])}</pre>"
        _tg_send(img_msg)

    footer = f"🔗 <b>Source (verify before posting):</b> {item['link']}"
    _tg_send(footer, disable_preview=False)


def send_multi_item_to_telegram(items, data):
    posts = data["posts"]
    is_thread = data.get("format") == "thread" and len(posts) > 1
    title = data.get("title", "").strip()

    kind_label = "Thread" if is_thread else "Single Post"
    title_part = f" · {_escape_html(title)}" if title else ""
    header = f"🧵 <b>New AI {kind_label} Draft</b>{title_part} ({len(posts)} post{'s' if len(posts) > 1 else ''})"
    _tg_send(header)

    for i, post in enumerate(posts, start=1):
        label = f"Post {i}/{len(posts)}" if is_thread else "Tap to copy:"
        _tg_send(f"{label}\n<pre>{_escape_html(post)}</pre>")

    if _is_true(data.get("needs_image")) and data.get("image_prompt"):
        img_msg = f"🎨 <b>Image Prompt (use with ChatGPT/Gemini/Nano Banana):</b>\n<pre>{_escape_html(data['image_prompt'])}</pre>"
        _tg_send(img_msg)

    lines = ["🔗 <b>Sources (verify before posting):</b>"]
    for it in items:
        lines.append(f"• {_escape_html(it['title'])}: {it['link']}")
    _tg_send("\n".join(lines), disable_preview=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# Rotates which category gets first pick each run, so high-volume sources (general
# "news" feeds always publish the most items) don't drown out lower-volume categories
# like tool/tutorial/innovation/offer just by sheer random-shuffle odds.
CATEGORY_SCHEDULE_BY_HOUR = {
    0: "news",
    4: "tool",
    8: "tutorial",
    12: "news",
    16: "innovation",
    20: "offer",
}

# Categories that get bundled into a multi-item listicle/thread instead of one
# single-tool post — matches the "5-25 tools in one post" style, and clears
# several backlogged items per run instead of just one.
MULTI_ITEM_CATEGORIES = {"tool", "tutorial"}
MULTI_ITEM_MAX = 6

# Two of the six daily slots lean toward thread format, so threads aren't left
# to pure chance (a single-post-biased prompt was producing zero threads/day).
THREAD_LEANING_HOURS = {4, 16}


def get_target_category():
    hour = datetime.datetime.utcnow().hour
    slot = (hour // 4) * 4
    return CATEGORY_SCHEDULE_BY_HOUR.get(slot, "news")


def get_format_directive():
    hour = datetime.datetime.utcnow().hour
    slot = (hour // 4) * 4
    if slot in THREAD_LEANING_HOURS:
        return (
            "\nFORMAT DIRECTIVE FOR THIS RUN: lean strongly toward format=thread. Actively look for "
            "a second and third angle (a limitation, a comparison, a practical implication, context "
            "the source gives) rather than settling for single. Only output single if there is truly "
            "nothing more to say beyond the first point."
        )
    return ""


def main():
    state = load_state()
    posted_hashes = {entry["hash"] for entry in state}

    target_category = get_target_category()
    all_candidates = collect_all_items()
    print(f"[info] fetched {len(all_candidates)} candidate items; target category this run: {target_category}")

    preferred = [c for c in all_candidates if c.get("category_hint") == target_category]
    other = [c for c in all_candidates if c.get("category_hint") != target_category]
    random.shuffle(preferred)
    random.shuffle(other)
    print(f"[info] {len(preferred)} candidates matched target category")

    # Tool/tutorial slots: bundle several fresh items into one listicle-style post
    # instead of spending the whole run on a single tool.
    if target_category in MULTI_ITEM_CATEGORIES:
        fresh_preferred = []
        seen_hashes = set()
        for c in preferred:
            if not c["title"] or not c["link"]:
                continue
            h = link_hash(c["link"])
            if h in posted_hashes or h in seen_hashes:
                continue
            c["_hash"] = h
            fresh_preferred.append(c)
            seen_hashes.add(h)
            if len(fresh_preferred) >= MULTI_ITEM_MAX:
                break

        if fresh_preferred:
            try:
                data = generate_multi_item_post(fresh_preferred, target_category)
                send_multi_item_to_telegram(fresh_preferred, data)
                for it in fresh_preferred:
                    state.append(
                        {"hash": it["_hash"], "link": it["link"], "title": it["title"], "ts": int(time.time())}
                    )
                save_state(state)
                print(f"[ok] posted multi-item {target_category} post covering {len(fresh_preferred)} candidates")
                return
            except Exception as e:
                print(f"[warn] multi-item flow failed, falling back to single-item flow: {e}")
        else:
            print(f"[info] no fresh {target_category} items available, falling back to single-item flow")

    # Single-item flow (news/innovation/offer, or fallback for tool/tutorial)
    candidates = preferred + other
    format_directive = get_format_directive()

    for item in candidates:
        h = link_hash(item["link"])
        if h in posted_hashes:
            continue

        try:
            thread_data = generate_thread(item, format_directive=format_directive)
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
