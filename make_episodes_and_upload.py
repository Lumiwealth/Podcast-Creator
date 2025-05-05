#!/usr/bin/env python3
"""
make_episodes_and_upload.py  –  end-to-end Transistor uploader

Workflow
────────
1. Read scripts/*.txt
2. Make (or reuse) audio/*.mp3 with OpenAI TTS
3. Authorise upload on Transistor, PUT to S3
4. POST /episodes   → draft episode
5. PATCH /episodes/:id/publish  (if “autopublish” in filename)

Env vars
────────
OPENAI_API_KEY
TRANSISTOR_API_KEY
TRANSISTOR_SHOW_ID
"""

import os, sys, json, logging, requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
TRANSISTOR_KEY  = os.getenv("TRANSISTOR_API_KEY")
TRANSISTOR_SHOW = os.getenv("TRANSISTOR_SHOW_ID")
if not (OPENAI_KEY and TRANSISTOR_KEY and TRANSISTOR_SHOW):
    sys.exit("❌  Set OPENAI_API_KEY, TRANSISTOR_API_KEY, TRANSISTOR_SHOW_ID")

TTS_ENDPOINT       = "https://api.openai.com/v1/audio/speech"
CHAT_ENDPOINT      = "https://api.openai.com/v1/chat/completions"
TX_ROOT            = "https://api.transistor.fm/v1"
AUTHORISE_ENDPOINT = f"{TX_ROOT}/episodes/authorize_upload"
EPISODE_ENDPOINT   = f"{TX_ROOT}/episodes"

SCRIPTS_DIR = Path("scripts")
AUDIO_DIR   = Path("audio")
MODEL_TTS   = "gpt-4o-mini-tts"
VOICE       = "alloy"
MODEL_CHAT  = "gpt-4o-mini"
CHUNK_SIZE  = 3_000
MAX_WORKERS = 4

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def dump_response(resp: requests.Response, label: str) -> None:
    """Always log URL, method, body (truncated), and response JSON/text."""
    try:
        logging.info("%s → %s %s", label, resp.status_code, resp.reason)
        req = resp.request
        logging.info("  %s %s", req.method, req.url)

        body = req.body
        if body is None:
            logging.info("  Sent: <None>")
        elif isinstance(body, (str, bytes)):
            logging.info("  Sent: %s", body[:500] + ("…" if len(body) > 500 else ""))
        else:
            logging.info("  Sent: <%s>", type(body).__name__)

        try:
            logging.info("  Rcvd: %s", json.dumps(resp.json(), indent=2)[:1000])
        except ValueError:
            logging.info("  Rcvd: %s", resp.text[:1000])
    except Exception:  # keep logs from ever crashing main flow
        logging.exception("dump_response failed but was suppressed")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int):
    return [text[i : i + size] for i in range(0, len(text), size)]

def tts_chunk(idx: int, chunk: str):
    r = requests.post(
        TTS_ENDPOINT,
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={"model": MODEL_TTS, "voice": VOICE, "format": "mp3", "input": chunk},
    )
    dump_response(r, f"TTS chunk {idx}")
    r.raise_for_status()
    return idx, r.content

def generate_description(title: str):
    r = requests.post(
        CHAT_ENDPOINT,
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={
            "model": MODEL_CHAT,
            "messages": [
                {
                    "role": "system",
                    "content": "You write concise, engaging podcast descriptions (max 2 sentences).",
                },
                {"role": "user", "content": f'Write a podcast description for "{title}".'},
            ],
            "max_tokens": 150,
            "temperature": 0.7,
        },
    )
    dump_response(r, "Chat description")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def get_existing_titles() -> set[str]:
    headers = {"x-api-key": TRANSISTOR_KEY}
    titles, page, per = set(), 1, 100
    while True:
        params = {
            "show_id": TRANSISTOR_SHOW,
            "pagination[page]": page,
            "pagination[per]": per,
        }
        r = requests.get(EPISODE_ENDPOINT, headers=headers, params=params)
        dump_response(r, f"Fetch existing page {page}")
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        titles.update(ep["attributes"]["title"] for ep in data)
        meta = r.json().get("meta", {})
        if page >= meta.get("totalPages", page):
            break
        page += 1
    return titles

# ──────────────────────────────────────────────────────────────────────────────
# Transistor upload
# ──────────────────────────────────────────────────────────────────────────────
def transistor_authorise(filename: str) -> tuple[str, str]:
    r = requests.get(
        AUTHORISE_ENDPOINT,
        headers={"x-api-key": TRANSISTOR_KEY},
        params={"filename": filename},
    )
    dump_response(r, "Authorize upload")
    r.raise_for_status()
    attrs = r.json()["data"]["attributes"]
    return attrs["upload_url"], attrs["audio_url"]

def transistor_put_audio(upload_url: str, mp3_path: Path):
    with mp3_path.open("rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": "audio/mpeg"})
    dump_response(r, "PUT MP3 → S3")
    r.raise_for_status()

def transistor_create_episode(title: str, description: str, audio_url: str) -> str:
    data = {
        "episode[show_id]": TRANSISTOR_SHOW,
        "episode[title]": title,
        "episode[description]": description,
        "episode[audio_url]": audio_url,
    }
    r = requests.post(EPISODE_ENDPOINT, headers={"x-api-key": TRANSISTOR_KEY}, data=data)
    dump_response(r, "Create episode")
    r.raise_for_status()
    ep_id = r.json()["data"]["id"]
    logging.info("✅  Episode %s created (draft)", ep_id)
    return ep_id

def transistor_publish_episode(ep_id: str):
    url  = f"{EPISODE_ENDPOINT}/{ep_id}/publish"
    data = {"episode[status]": "published"}
    r = requests.patch(url, headers={"x-api-key": TRANSISTOR_KEY}, data=data)
    dump_response(r, "Publish episode")
    r.raise_for_status()
    logging.info("🚀  Episode %s published", ep_id)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    AUDIO_DIR.mkdir(exist_ok=True)
    existing = get_existing_titles()

    for script_path in sorted(SCRIPTS_DIR.glob("*.txt")):
        base      = script_path.stem
        title     = base.split("_", 1)[1].replace("_", " ")
        mp3_path  = AUDIO_DIR / f"{base}.mp3"

        logging.info("──── %s ────", title)

        # 1. TTS (cached)
        if not mp3_path.exists():
            text  = script_path.read_text(encoding="utf-8")
            parts = [None] * len(chunk_text(text, CHUNK_SIZE))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futs = {
                    pool.submit(tts_chunk, i, chunk): i
                    for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE))
                }
                for fut in as_completed(futs):
                    i, audio = fut.result()
                    parts[i] = audio
            with mp3_path.open("wb") as out:
                for seg in parts:
                    out.write(seg)
            logging.info("MP3 written → %s", mp3_path)
        else:
            logging.info("MP3 cached   → %s", mp3_path)

        # 2. Skip if already on Transistor
        if title in existing:
            logging.info("Episode already exists on Transistor – skipping")
            continue

        # 3. Description
        desc = generate_description(title)

        # 4. Upload + episode
        try:
            upload_url, audio_url = transistor_authorise(mp3_path.name)
            transistor_put_audio(upload_url, mp3_path)
            ep_id = transistor_create_episode(title, desc, audio_url)

            # 5. Autopublish?
            if "autopublish" in base.lower():
                transistor_publish_episode(ep_id)

        except Exception as e:
            logging.exception("❌  Failed to publish %s: %s", title, e)

if __name__ == "__main__":
    logging.info("=== Podcast batch started %s ===", datetime.now())
    try:
        main()
    finally:
        logging.info("=== Podcast batch finished %s ===", datetime.now())