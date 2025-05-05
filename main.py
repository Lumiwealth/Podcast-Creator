# web_server.py – Flask app to brainstorm, iterate, and publish podcast episodes
# ==========================================================================
# Quick start on Replit (or any Python 3.9+):
#   1. Place this file next to **make_episodes_and_upload.py** (the uploader).
#   2. `pip install flask openai requests` (Flask & OpenAI client).
#   3. Set env vars: OPENAI_API_KEY, TRANSISTOR_API_KEY, TRANSISTOR_SHOW_ID.
#
# The flow
#   • Home page: input *only* the episode idea; server auto‑generates a title & script.
#   • Review page: editable title & script, optional feedback box to regenerate.
#   • Approve → script → TTS MP3 → Transistor draft episode (toggle publish).
#   • Drafts are held in‑memory; switch to DB/Redis for production use.
# ---------------------------------------------------------------------------

import os, uuid
from pathlib import Path
from typing import Dict

from flask import Flask, render_template, request, redirect, url_for, flash
from openai import OpenAI

# Import helpers from the finished uploader script
import make_episodes_and_upload as uploader

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)
PREFERRED_MODEL = "gpt-4"   # best creative model as of May 2025
FALLBACK_MODEL  = "gpt-3.5-turbo"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
app.template_folder = "templates" # added template folder

# Ephemeral draft storage
DRAFTS: Dict[str, Dict] = {}

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI helpers
# ─────────────────────────────────────────────────────────────────────────────

def chat_create(messages, **kw):
    """Try preferred model, fall back if unavailable."""
    try:
        completion = client.chat.completions.create(
            model=PREFERRED_MODEL,
            messages=messages,
            **kw
        )
        return completion
    except Exception:
        completion = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            **kw
        )
        return completion

def generate_title(idea: str) -> str:
    messages = [
        {"role": "system", "content": "Create a catchy podcast episode title (≤10 words). Respond ONLY with the title."},
        {"role": "user", "content": idea},
    ]
    resp = chat_create(messages, temperature=0.9, max_tokens=32)
    return resp.choices[0].message.content.strip().strip("\"")

def generate_script(title: str, idea: str) -> str:
    messages = [
        {"role": "system", "content": """You are a master storyteller and podcast script writer in the style of Malcolm Gladwell. Write a pure narrative script:

- Begin with a compelling hook or unexpected anecdote
- Weave together multiple narratives that build to larger insights
- Use natural, conversational language optimized for text-to-speech
- Create pacing through sentence structure (NOT through audio directions)
- Write ONLY narrative text - no audio cues, music notes, pauses, or sound effects
- DO NOT include any markers like [Pause], [Music], [Sound effect], etc.
- Create emotional engagement through storytelling, not audio directions
- End with a powerful conclusion that ties everything together
- Use short paragraphs with clear transitions
- Write for the ear - no citations, parentheticals, or formatting

Write a ~15000 word pure narrative script with NO audio direction markers."""}, #Increased word count
        {"role": "user", "content": f"Title: {title}\n\nCore idea: {idea}\n\nCreate a compelling podcast episode that explores this topic through vivid storytelling and unexpected connections, building to a powerful insight."},
    ]
    resp = chat_create(messages, temperature=0.7, max_tokens=10000)
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/generate", methods=["POST"])
def generate():
    idea = request.form["idea"].strip()
    title  = generate_title(idea)
    script = generate_script(title, idea)
    draft_id = str(uuid.uuid4())
    DRAFTS[draft_id] = {"idea": idea, "title": title, "script": script}
    return redirect(url_for("review", draft_id=draft_id))

@app.route("/review/<draft_id>")
def review(draft_id):
    d = DRAFTS.get(draft_id)
    if not d:
        flash("Draft not found")
        return redirect(url_for("home"))
    return render_template("review.html", draft_id=draft_id, title=d['title'], script=d['script'])

@app.route("/revise/<draft_id>", methods=["POST"])
def revise(draft_id):
    d = DRAFTS.get(draft_id)
    if not d:
        flash("Draft not found")
        return redirect(url_for("home"))

    d["title"]  = request.form.get("title", d["title"]).strip()
    d["script"] = request.form.get("script", d["script"]).strip()
    feedback     = request.form.get("feedback", "").strip()

    if not feedback:
        flash("Provide feedback or edit script, then Approve to publish.")
        return redirect(url_for("review", draft_id=draft_id))

    messages = [
        {"role": "system", "content": "Improve the podcast script per feedback (~1200 words)."},
        {"role": "assistant", "content": d["script"]},
        {"role": "user", "content": feedback},
    ]
    resp = chat_create(messages, temperature=0.7, max_tokens=2048)
    d["script"] = resp.choices[0].message.content.strip()
    return redirect(url_for("review", draft_id=draft_id))

@app.route("/generate_mp3/<draft_id>", methods=["POST"])
def generate_mp3(draft_id):
    d = DRAFTS.get(draft_id)
    if not d:
        flash("Draft not found")
        return redirect(url_for("home"))

    title = request.form.get("title", d["title"]).strip()
    script = request.form.get("script", d["script"]).strip()
    d["title"] = title
    d["script"] = script

    # Generate MP3
    chunks = uploader.chunk_text(script, uploader.CHUNK_SIZE)
    audio_parts = [uploader.tts_chunk(i, c)[1] for i, c in enumerate(chunks)]
    #Improved filename generation
    unique_id = str(uuid.uuid4())[:4]
    mp3_path = Path("audio") / f"{title.replace(' ', '_')}_{unique_id}.mp3"
    mp3_path.parent.mkdir(exist_ok=True)
    with mp3_path.open("wb") as f:
        for part in audio_parts:
            f.write(part)

    # Store MP3 path in draft
    d["mp3_path"] = str(mp3_path)
    flash("MP3 generated! You can now approve and upload when ready.")
    return redirect(url_for("review", draft_id=draft_id))

@app.route("/approve/<draft_id>", methods=["POST"])
def approve(draft_id):
    d = DRAFTS.pop(draft_id, None)
    if not d:
        flash("Draft not found")
        return redirect(url_for("home"))

    title  = request.form.get("title", d["title"]).strip()
    script = request.form.get("script", d["script"]).strip()

    # 1. Synthesize MP3 via uploader helpers
    chunks = uploader.chunk_text(script, uploader.CHUNK_SIZE)
    audio_parts = [uploader.tts_chunk(i, c)[1] for i, c in enumerate(chunks)]
    mp3_path = Path("audio") / f"manual_{uuid.uuid4()}.mp3"
    mp3_path.parent.mkdir(exist_ok=True)
    with mp3_path.open("wb") as f:
        for part in audio_parts:
            f.write(part)

    # 2. Upload to Transistor (draft)
    up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
    uploader.transistor_put_audio(up_url, mp3_path)
    ep_id = uploader.transistor_create_episode(title, script, audio_url)

    # Uncomment to auto‑publish immediately
    # uploader.transistor_publish_episode(ep_id)

    flash(f"Episode drafted on Transistor (ID {ep_id}).")
    return redirect(url_for("home"))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)