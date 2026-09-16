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
- NEVER use em dashes (—) or double hyphens (--) anywhere in any post. This is one of the most
  obvious AI-writing tells and readers notice it immediately. Use a period, a comma, or simply split
  into two sentences instead.
- You may use a single line break (\n) inside a post's text when it genuinely improves readability
  (e.g. separating a hook line from the supporting detail, or setting a punchline apart) — this is
  normal, human formatting on X. Don't force it into every post, but don't avoid it either; a wall of
  unbroken text across every single post is itself a tell.
- Post 1 (or the single post) must have a hook in the first 8-10 words that makes someone stop
  scrolling. Use a specific number, surprising fact, or bold claim from the source — not a vague
  generic opener. NEVER open with throat-clearing like "Lately I've been...", "I've been digging
  into...", "So I found...", "Here's something interesting" — get straight to the fact/number/claim.
- If it's a thread, EVERY post must add NEW information or a NEW angle. Never restate or rephrase a
  point already made in an earlier post. Each item's post should include a one-line reaction or
  "why this matters" angle — not just a reworded restatement of the source title/summary.
- If you can't come up with a genuinely new angle, cut the thread short instead of padding it.
- Do NOT end with filler like "stay tuned", "more insights soon", "what a time to be alive". End with
  either: a sharp question to the audience, a contrarian/hot take, or one concrete practical takeaway.
- Each post must be under 270 characters (X limit is 280, leave buffer).
- Use 1-2 relevant emojis per post where they add punch (not every sentence, not decorative spam,
  not the same emoji every time — pick ones that fit the specific content).
- Include 1-3 relevant hashtags in the LAST post — prefer broad, high-traffic tags people actually
  follow/search (#AI, #AITools, #MachineLearning, #TechNews, #ChatGPT) over narrow or redundant
  combinations (avoid pairing near-duplicates like "#AIML #ML" together). Hashtags render as
  clickable blue links on X and help visibility even though they don't drive the algorithm much.
- Plain, direct, confident tone. No corporate hedging language ("it is fascinating to see", "highlights").
- IMAGE DECISION: Set needs_image true whenever a visual would plausibly boost engagement — this
  includes most tool/product posts (show the tool's interface/concept), most tutorial posts (show the
  technique/result), and news/innovation posts with a concrete visual concept (a robot, a chart-like
  idea, a before/after). Only set needs_image false when the content is genuinely abstract with
  nothing visual to depict (e.g. a pure policy debate, an abstract opinion take with no concrete
  imagery). When in doubt, default to true — err toward including an image prompt, not skipping it.
- IMAGE STYLE — think a well-designed PowerPoint/slide-CARD with a bold hook, not an X (Twitter) post
  image built like a YouTube video thumbnail. The image needs a bold, catchy HEADLINE/hook text as its
  visual anchor, supported by clean icons, short callout labels, or a simple layout — a hero photo or
  illustrated character is OPTIONAL, not required. Good variations include: a pure text+icon card
  (bold headline + a few supporting icons, no photo/character at all), a clean grid/list layout (like
  a "top N tools" card with an icon per item), or occasionally a fun illustrated character/mascot when
  the topic suits humor. AVOID multi-panel numbered "how it works" instructional diagrams with small
  step-by-step boxes — those read as documentation/tutorial slides, not a hook-driven post image. The
  headline text must be BIG, bold, and high-contrast — not a small label buried among icons.
- IMAGE TEXT IS MANDATORY AND SPECIFIC: the image_prompt MUST spell out the EXACT headline text (in
  quotes) that should render on the image, and that text must be specific to THIS story/post — never
  leave it to the image generator to invent its own caption. A vague instruction like "bold readable
  text" is NOT enough and will produce an off-topic image. Example of the right level of detail:
  image_prompt includes something like: The headline text reading exactly "4 AI TUTORIALS NOBODY'S
  COVERING" in bold white and cyan letters, plus a small supporting line "embeddings · Gradio ·
  fine-tuning · benchmarks" — derive the exact wording from THIS post's actual hook/topic, not a
  generic placeholder.
- IMAGE FONT: always name a specific bold display-style font treatment in the prompt, not a plain
  default sans-serif — e.g. "bold condensed uppercase display font like Anton or Bebas Neue", "thick
  rounded sans like Poppins Bold", "a punchy marker/brush-stroke style for the key word", or "chunky
   3D-effect lettering with a drop shadow". A plain thin generic font is a common reason these images
  look flat — call out the exact typographic feel you want.
- IMAGE VARIETY — IMPORTANT: do NOT default to the same dark-navy-blue tech-dashboard look every time.
  Vary the color palette and hero visual each time (different background colors, different hero
  scene/character, different accent colors) — a bot that always posts a visually identical template
  becomes obviously recognizable as automated, which hurts the account.

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
- NEVER use em dashes (—) or double hyphens (--) anywhere — a well-known AI-writing tell. Use a
  period, comma, or a new sentence instead. A single \n line break inside a post is fine when it
  helps readability (e.g. separating the hook from the detail).
- Post 1 must open with a hook in the first 8-10 words — a number, surprising fact, or bold claim.
  NEVER open with throat-clearing like "Lately I've been...", "I've been digging into...", "So I
  found..." — get straight to the point.
- If covering multiple items: come up with a bold, catchy TITLE for this post/thread (its own field,
  not inside the posts) that matches the actual hook used in post 1 — they should feel connected, not
  like two different taglines. Then one post per item you kept — say what it is and give a one-line
  reaction/why-it-matters, based ONLY on the title/context given, don't invent specifics, don't just
  reword the source title. Final post ends with a question or hot take + 1-3 broad, high-traffic
  hashtags (#AI, #AITools, #MachineLearning — avoid redundant pairs like "#AIML #ML" together).
- If only 1 item is genuinely worth it: leave title empty, write normal single/thread content with the
  same hook standards as usual.
- Never copy titles verbatim — rewrite naturally. No clichéd AI phrases ("game-changer", "unlock the
  power of"). Vary sentence rhythm. Have a point of view, don't just list facts.
- 1-2 relevant emojis per post (not spammy). Each post under 270 characters.
- IMAGE: set needs_image true when a visual would help (default true for tool/multi-item content).
  Think a well-designed PowerPoint/slide-CARD with a bold hook headline as the visual anchor, supported
  by clean icons or short labels — a hero photo or character is optional, not required. A pure
  text+icon card (bold headline + a few icons, no photo at all) is a perfectly good option, as is a
  clean grid/list layout naming each item with an icon. Avoid multi-panel numbered "how it works"
  instructional diagrams — those read as tutorial slides, not a hook-driven post. Headline text should
  be big, bold, high-contrast — not a small label. Vary the color palette and layout each time.
- IMAGE TEXT IS MANDATORY AND SPECIFIC: image_prompt MUST spell out the EXACT headline text (in
  quotes) to render, matching the title/hook of THIS post — never leave the wording to the image
  generator, or it will invent an unrelated caption. Include the specific items/theme in a short
  supporting line if it fits (e.g. names of the tools covered), not a generic placeholder.
- IMAGE FONT: name a specific bold display-style font treatment, not a plain default sans-serif — e.g.
  "bold condensed uppercase like Anton or Bebas Neue", "chunky rounded sans like Poppins Bold", or
  "punchy marker/brush-stroke accent for the key word". A plain thin font reads flat and generic.

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


def filter_kept_items(items, dropped_titles):
    """
    The multi-item prompt may drop weak items from the actual posts while we
    still passed it the full candidate list. Match on normalized title text so
    the Telegram source list only shows items actually covered, and so dropped
    items stay unposted (available to be reconsidered another day) instead of
    being silently burned from the pool.
    """
    dropped_norm = {str(t).strip().lower() for t in (dropped_titles or [])}
    kept = [it for it in items if it["title"].strip().lower() not in dropped_norm]
    dropped = [it for it in items if it["title"].strip().lower() in dropped_norm]
    return kept, dropped


def send_multi_item_to_telegram(items, data, category="tool"):
    posts = data["posts"]
    is_thread = data.get("format") == "thread" and len(posts) > 1
    title = data.get("title", "").strip()

    kind_label = "Thread" if is_thread else "Single Post"
    title_part = f" · {_escape_html(title)}" if title else ""
    header = (
        f"🧵 <b>New AI {kind_label} Draft</b> · #{category.upper()}{title_part} "
        f"({len(posts)} post{'s' if len(posts) > 1 else ''})"
    )
    _tg_send(header)

    for i, post in enumerate(posts, start=1):
        label = f"Post {i}/{len(posts)}" if is_thread else "Tap to copy:"
        _tg_send(f"{label}\n<pre>{_escape_html(post)}</pre>")

    if _is_true(data.get("needs_image")) and data.get("image_prompt"):
        img_msg = f"🎨 <b>Image Prompt (use with ChatGPT/Gemini/Nano Banana):</b>\n<pre>{_escape_html(data['image_prompt'])}</pre>"
        _tg_send(img_msg)

    # Only list items actually covered in the posts — dropped candidates are
    # excluded here so the link list matches what the thread actually discusses.
    kept, _ = filter_kept_items(items, data.get("dropped_titles"))
    lines = ["🔗 <b>Sources (verify before posting):</b>"]
    for it in kept:
        lines.append(f"• {_escape_html(it['title'])}: {it['link']}")
    _tg_send("\n".join(lines), disable_preview=True)


# ---------------------------------------------------------------------------
# AI TOOLS DIRECTORY POST — pre-vetted official links, category-grouped, image-led
# ---------------------------------------------------------------------------
# These are stable, well-known products with stable official URLs — hard-coded
# here (not LLM-generated) so links are never hallucinated. Update this list
# occasionally as tools rise/fall in relevance; the LLM only picks phrasing,
# never invents a tool name or URL.
TOOL_DIRECTORY = {
    "Chatbots & Assistants": [
        ("ChatGPT", "https://chatgpt.com"),
        ("Claude", "https://claude.ai"),
        ("Gemini", "https://gemini.google.com"),
        ("Grok", "https://grok.com"),
        ("Perplexity", "https://www.perplexity.ai"),
    ],
    "Writing": [
        ("Jasper", "https://www.jasper.ai"),
        ("Copy.ai", "https://www.copy.ai"),
        ("Writesonic", "https://writesonic.com"),
        ("Grammarly", "https://www.grammarly.com"),
        ("Notion AI", "https://www.notion.so/product/ai"),
    ],
    "Image Generation": [
        ("Midjourney", "https://www.midjourney.com"),
        ("DALL-E", "https://openai.com/dall-e-3"),
        ("Stable Diffusion", "https://stability.ai"),
        ("Ideogram", "https://ideogram.ai"),
        ("Leonardo AI", "https://leonardo.ai"),
    ],
    "Video Generation": [
        ("Sora", "https://sora.chatgpt.com"),
        ("Runway", "https://runwayml.com"),
        ("Pika", "https://pika.art"),
        ("Synthesia", "https://www.synthesia.io"),
        ("HeyGen", "https://www.heygen.com"),
        ("Kling", "https://klingai.com"),
    ],
    "Music & Audio": [
        ("Suno", "https://suno.com"),
        ("Udio", "https://www.udio.com"),
        ("ElevenLabs", "https://elevenlabs.io"),
    ],
    "Design": [
        ("Canva", "https://www.canva.com"),
        ("Figma", "https://www.figma.com"),
        ("Adobe Firefly", "https://firefly.adobe.com"),
    ],
    "Coding": [
        ("GitHub Copilot", "https://github.com/features/copilot"),
        ("Cursor", "https://www.cursor.com"),
        ("Replit", "https://replit.com"),
        ("Claude Code", "https://claude.com/claude-code"),
    ],
    "Automation": [
        ("Zapier", "https://zapier.com"),
        ("Make", "https://www.make.com"),
        ("n8n", "https://n8n.io"),
    ],
    "Websites": [
        ("Webflow", "https://webflow.com"),
        ("Wix Studio", "https://www.wix.com/studio"),
        ("Framer", "https://www.framer.com"),
    ],
    "Marketing": [
        ("HubSpot AI", "https://www.hubspot.com/products/marketing/ai"),
        ("AdCreative.ai", "https://www.adcreative.ai"),
        ("Predis.ai", "https://predis.ai"),
    ],
}

DIRECTORY_PROMPT = """You are creating an "AI tools directory" post for an X (Twitter) account in the AI niche.

This is a well-known content format: a short, punchy CAPTION tweet paired with an eye-catching
GRID/CARD graphic image that lists curated tools grouped by category — the kind of image people
screenshot and save. The value lives in the image, not a long thread. The tools below are real and
pre-vetted — use them exactly as given, never invent a tool name.

Categories and tools for this post:
{items_block}

Write:
- NEVER use em dashes (—) or double hyphens (--) anywhere — a well-known AI-writing tell.
- ONE short caption post, under 270 characters: a strong hook in the first 8-10 words (no throat-
  clearing like "Here's a list of..."), plus 1-3 broad hashtags at the end (#AI, #AITools). The
  caption earns the click/save; it does not need to name every tool.
- An image_prompt for a clean GRID/CARD graphic: a bold headline at the top matching the caption's
  hook (spell out the EXACT headline text in quotes), then the tools organized by the categories
  given, one short icon or bullet per tool name (use the exact names given, do not invent more).
  Presentation/poster style, high contrast, a specific bold display font named explicitly (e.g. "bold
  condensed uppercase like Anton or Bebas Neue"). Pure text+icon grid is exactly right here, no hero
  photo needed.
- No clichéd AI phrases. Have a point of view, not generic filler like "you don't want to miss these".

Return STRICT JSON only, no markdown fences:
{{
  "post": "the short caption text",
  "image_prompt": "detailed grid/card image prompt with the exact headline text spelled out in quotes"
}}
"""


def build_tool_directory_selection(num_categories=6, tools_per_category=3):
    categories = random.sample(list(TOOL_DIRECTORY.keys()), k=min(num_categories, len(TOOL_DIRECTORY)))
    selection = {}
    for cat in categories:
        pool = TOOL_DIRECTORY[cat]
        selection[cat] = random.sample(pool, k=min(tools_per_category, len(pool)))
    return selection


def generate_directory_post(selection):
    items_block = "\n".join(
        f"{cat}: " + ", ".join(name for name, _ in tools) for cat, tools in selection.items()
    )
    prompt = DIRECTORY_PROMPT.format(items_block=items_block)
    return call_llm(prompt, temperature=0.8)


def send_directory_post_to_telegram(selection, data):
    header = "🧵 <b>New AI Tools Directory Post</b> (image-led)"
    _tg_send(header)
    _tg_send(f"Tap to copy:\n<pre>{_escape_html(data['post'])}</pre>")

    if data.get("image_prompt"):
        _tg_send(f"🎨 <b>Image Prompt (use with ChatGPT/Gemini/Nano Banana):</b>\n<pre>{_escape_html(data['image_prompt'])}</pre>")

    lines = ["🔗 <b>Official links (for your reference / optional reply thread):</b>"]
    for cat, tools in selection.items():
        lines.append(f"<b>{_escape_html(cat)}</b>")
        for name, url in tools:
            lines.append(f"• {_escape_html(name)}: {url}")
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
    12: "directory",
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
    print(f"[info] target category this run: {target_category}")

    # Directory posts are self-contained (pre-vetted tool list, no RSS needed).
    if target_category == "directory":
        try:
            selection = build_tool_directory_selection()
            data = generate_directory_post(selection)
            send_directory_post_to_telegram(selection, data)
            print(f"[ok] posted AI tools directory post covering {sum(len(v) for v in selection.values())} tools")
            return
        except Exception as e:
            print(f"[warn] directory flow failed, falling back to news single-item flow: {e}")
            target_category = "news"

    all_candidates = collect_all_items()
    print(f"[info] fetched {len(all_candidates)} candidate items")

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
                send_multi_item_to_telegram(fresh_preferred, data, category=target_category)
                kept, dropped = filter_kept_items(fresh_preferred, data.get("dropped_titles"))
                for it in kept:
                    state.append(
                        {"hash": it["_hash"], "link": it["link"], "title": it["title"], "ts": int(time.time())}
                    )
                save_state(state)
                if dropped:
                    print(f"[info] {len(dropped)} candidate(s) dropped this run, left available for later")
                print(f"[ok] posted multi-item {target_category} post covering {len(kept)} kept item(s)")
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
