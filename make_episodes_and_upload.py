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
MODEL_TTS   = "tts-1" # Standard, fast TTS model
VOICE       = "alloy"
MODEL_CHAT  = "gpt-4o-mini" # Suitable for short description generation
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
    """Log URL, method, truncated body, and truncated response."""
    try:
        logging.info("%s → %s %s", label, resp.status_code, resp.reason)
        req = resp.request
        logging.info("  %s %s", req.method, req.url)

        body = req.body
        body_repr = "<None>"
        if body is not None:
            try:
                # Attempt to decode if bytes, otherwise use as is if str
                body_str = body.decode('utf-8') if isinstance(body, bytes) else body
                trunc_len = 200 # Reduced truncation length
                body_repr = body_str[:trunc_len] + ("…" if len(body_str) > trunc_len else "")
            except UnicodeDecodeError:
                body_repr = f"<Bytes body, len={len(body)}, decode failed>"
            except Exception: # Catch other potential issues
                 body_repr = f"<{type(body).__name__} body, error processing>"

        logging.info("  Sent: %s", body_repr)

        try:
            # Also truncate response JSON/text more aggressively
            resp_trunc_len = 200
            resp_json_str = json.dumps(resp.json(), indent=2)
            logging.info("  Rcvd: %s", resp_json_str[:resp_trunc_len] + ("…" if len(resp_json_str) > resp_trunc_len else ""))
        except ValueError: # If not JSON
            resp_text = resp.text
            logging.info("  Rcvd: %s", resp_text[:resp_trunc_len] + ("…" if len(resp_text) > resp_trunc_len else ""))
    except Exception:
        logging.exception("dump_response failed but was suppressed")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int):
    logging.info(f"Chunking text of length {len(text)} with chunk size {size}")
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    logging.info(f"Chunked into {len(chunks)} parts")
    return chunks

def tts_chunk(idx: int, chunk: str):
    logging.info(f"Requesting TTS for chunk {idx} (length {len(chunk)})")
    r = requests.post(
        TTS_ENDPOINT,
        headers={"Authorization": f"Bearer {OPENAI_KEY}"},
        json={"model": MODEL_TTS, "voice": VOICE, "format": "mp3", "input": chunk},
    )
    dump_response(r, f"TTS chunk {idx}")
    r.raise_for_status()
    logging.info(f"TTS chunk {idx} received, {len(r.content)} bytes")
    return idx, r.content

def generate_description(title: str):
    logging.info(f"Generating description for title: {title}")
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
    desc = r.json()["choices"][0]["message"]["content"].strip()
    logging.info(f"Description generated: {desc}")
    return desc

def get_existing_titles() -> set[str]:
    logging.info("Fetching existing episode titles from Transistor")
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
            logging.info(f"No more data on page {page}")
            break
        titles.update(ep["attributes"]["title"] for ep in data)
        meta = r.json().get("meta", {})
        logging.info(f"Fetched page {page}, got {len(data)} episodes")
        if page >= meta.get("totalPages", page):
            break
        page += 1
    logging.info(f"Total existing titles fetched: {len(titles)}")
    return titles

# ──────────────────────────────────────────────────────────────────────────────
# Transistor upload
# ──────────────────────────────────────────────────────────────────────────────
def transistor_authorise(filename: str) -> tuple[str, str]:
    logging.info(f"Authorizing upload for filename: {filename}")
    r = requests.get(
        AUTHORISE_ENDPOINT,
        headers={"x-api-key": TRANSISTOR_KEY},
        params={"filename": filename},
    )
    dump_response(r, "Authorize upload")
    r.raise_for_status()
    attrs = r.json()["data"]["attributes"]
    logging.info(f"Authorized upload: upload_url and audio_url received")
    return attrs["upload_url"], attrs["audio_url"]

def transistor_put_audio(upload_url: str, mp3_path: Path):
    logging.info(f"Uploading MP3 file to S3: {mp3_path}")
    with mp3_path.open("rb") as f:
        r = requests.put(upload_url, data=f, headers={"Content-Type": "audio/mpeg"})
    dump_response(r, "PUT MP3 → S3")
    r.raise_for_status()
    logging.info("MP3 upload to S3 complete")

def transistor_create_episode(title: str, description: str, audio_url: str) -> str:
    logging.info(f"Creating episode draft on Transistor: {title}")
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
    logging.info(f"✅  Episode {ep_id} created (draft)")
    return ep_id

def transistor_publish_episode(ep_id: str):
    logging.info(f"Publishing episode {ep_id} on Transistor")
    url  = f"{EPISODE_ENDPOINT}/{ep_id}/publish"
    data = {"episode[status]": "published"}
    r = requests.patch(url, headers={"x-api-key": TRANSISTOR_KEY}, data=data)
    dump_response(r, "Publish episode")
    r.raise_for_status()
    logging.info(f"🚀  Episode {ep_id} published")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    logging.info("Starting main() in make_episodes_and_upload.py")
    AUDIO_DIR.mkdir(exist_ok=True)
    existing = get_existing_titles()

    for script_path in sorted(SCRIPTS_DIR.glob("*.txt")):
        base      = script_path.stem
        title     = base.split("_", 1)[1].replace("_", " ")
        mp3_path  = AUDIO_DIR / f"{base}.mp3"

        logging.info(f"──── Processing: {title} ────")

        # 1. TTS (cached)
        if not mp3_path.exists():
            logging.info(f"MP3 does not exist, generating: {mp3_path}")
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
            logging.info(f"MP3 written → {mp3_path}")
        else:
            logging.info(f"MP3 cached   → {mp3_path}")

        # 2. Skip if already on Transistor
        if title in existing:
            logging.info("Episode already exists on Transistor – skipping")
            continue

        # 3. Description
        desc = generate_description(title)

        # 4. Upload + episode
        try:
            logging.info(f"Uploading and creating episode for: {title}")
            upload_url, audio_url = transistor_authorise(mp3_path.name)
            transistor_put_audio(upload_url, mp3_path)
            ep_id = transistor_create_episode(title, desc, audio_url)

            # 5. Autopublish?
            if "autopublish" in base.lower():
                logging.info(f"Autopublishing episode: {ep_id}")
                transistor_publish_episode(ep_id)

        except Exception as e:
            logging.exception(f"❌  Failed to publish {title}: {e}")

if __name__ == "__main__":
    logging.info("=== Podcast batch started %s ===", datetime.now())
    try:
        main()
    finally:
        logging.info("=== Podcast batch finished %s ===", datetime.now())