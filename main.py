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

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
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

def extract_title_from_script(script: str) -> str:
    # Look for the first line that could be a title
    lines = script.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line and len(line) <= 100 and not line.startswith(('A:', 'Q:', '-', '#', '>')):
            return line
    return "Untitled Episode"  # Fallback

def generate_script(idea: str) -> str:
    messages = [
        {"role": "system", "content": """You are a master storyteller and podcast script writer. Write a pure narrative script:

- Begin with a clear, catchy title on the first line
- Follow with a compelling hook or unexpected anecdote
- Weave together multiple narratives that build to larger insights
- Use natural, conversational language optimized for text-to-speech
- Create pacing through sentence structure (NOT through audio directions)
- Write ONLY narrative text - no audio cues, music notes, pauses, or sound effects
- DO NOT include any markers like [Pause], [Music], [Sound effect], etc.
- Create emotional engagement through storytelling, not audio directions
- End with a powerful conclusion that ties everything together
- Use short paragraphs with clear transitions
- Write for the ear - no citations, parentheticals, or formatting
- Write in an engaging narrative style without referencing specific authors

Write a ~15000 word pure narrative script with NO audio direction markers. Make sure it's engaging and compelling, but also long enough to be a full episode."""},
        {"role": "user", "content": f"Core idea: {idea}\n\nCreate a compelling podcast episode that explores this topic through vivid storytelling and unexpected connections, building to a powerful insight."}
    ]
    resp = chat_create(messages, temperature=0.7, max_tokens=4096)
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
    script = generate_script(idea)
    title = extract_title_from_script(script)
    draft_id = str(uuid.uuid4())
    DRAFTS[draft_id] = {"idea": idea, "title": title, "script": script}
    return redirect(url_for("review", draft_id=draft_id))

@app.route("/generate_and_publish", methods=["POST"])
def generate_and_publish():
    # Generate content
    idea = request.form["idea"].strip()
    script = generate_script(idea)
    title = extract_title_from_script(script)
    
    # Generate MP3
    chunks = uploader.chunk_text(script, uploader.CHUNK_SIZE)
    audio_parts = [uploader.tts_chunk(i, c)[1] for i, c in enumerate(chunks)]
    unique_id = str(uuid.uuid4())[:4]
    mp3_path = Path("audio") / f"{title.replace(' ', '_')}_{unique_id}.mp3"
    mp3_path.parent.mkdir(exist_ok=True)
    with mp3_path.open("wb") as f:
        for part in audio_parts:
            f.write(part)

    # Upload to Transistor and publish
    up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
    uploader.transistor_put_audio(up_url, mp3_path)
    ep_id = uploader.transistor_create_episode(title, script, audio_url)
    uploader.transistor_publish_episode(ep_id)  # Actually publish the episode

    flash(f'Episode published on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">View your published episode</a>')
    return redirect(url_for("home"))

@app.route("/review/<draft_id>")
def review(draft_id):
    d = DRAFTS.get(draft_id)
    if not d:
        flash("Draft not found")
        return redirect(url_for("home"))
    return render_template("review.html", draft_id=draft_id, title=d['title'], script=d['script'], draft=d)

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory("audio", filename)

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
    d["mp3_filename"] = mp3_path.name
    flash("MP3 generated successfully!")
    return redirect(url_for("review", draft_id=draft_id, mp3_ready=True))

@app.route("/upload/<draft_id>", methods=["POST"])
def upload(draft_id):
    d = DRAFTS.get(draft_id)
    if not d or "mp3_path" not in d:
        flash("MP3 not found")
        return redirect(url_for("review", draft_id=draft_id))

    # Upload to Transistor
    title = request.form.get("title", d["title"]).strip()
    script = request.form.get("script", d["script"]).strip()
    mp3_path = Path(d["mp3_path"])

    up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
    uploader.transistor_put_audio(up_url, mp3_path)
    ep_id = uploader.transistor_create_episode(title, script, audio_url)

    flash(f'Episode drafted on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">Publish your episode here</a>')
    return redirect(url_for("home"))

@app.route("/generate_and_upload/<draft_id>", methods=["POST"]) 
def generate_and_upload(draft_id):
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
    unique_id = str(uuid.uuid4())[:4]
    mp3_path = Path("audio") / f"{title.replace(' ', '_')}_{unique_id}.mp3"
    mp3_path.parent.mkdir(exist_ok=True)
    with mp3_path.open("wb") as f:
        for part in audio_parts:
            f.write(part)

    # Upload to Transistor
    up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
    uploader.transistor_put_audio(up_url, mp3_path)
    ep_id = uploader.transistor_create_episode(title, script, audio_url)

    flash(f'Episode drafted on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">Publish your episode here</a>')
    return redirect(url_for("home"))

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

    flash(f'Episode drafted on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">Publish your episode here</a>')
    return redirect(url_for("home"))

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)