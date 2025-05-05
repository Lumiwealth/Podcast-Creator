
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

from flask import Flask, render_template_string, request, redirect, url_for, flash
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

# Ephemeral draft storage
DRAFTS: Dict[str, Dict] = {}

# ─────────────────────────────────────────────────────────────────────────────
# HTML wrapper (Bootstrap 5)
# ─────────────────────────────────────────────────────────────────────────────
BASE_HTML = """<!doctype html>
<html lang='en'>
  <head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>Podcast Episode Builder</title>
    <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' rel='stylesheet'>
  </head>
  <body class='bg-light'>
    <div class='container py-5'>
      <h1 class='mb-4 text-center'>Podcast Episode Builder</h1>
      {% with msgs = get_flashed_messages() %}{% if msgs %}
        <div class='alert alert-info'>{{ msgs[0] }}</div>
      {% endif %}{% endwith %}
      {{ body|safe }}
    </div>
    <script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js'></script>
  </body>
</html>"""

def page(content: str):
    return render_template_string(BASE_HTML, body=content)

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
        {"role": "system", "content": """You are a master storyteller and podcast script writer in the style of Malcolm Gladwell. Your scripts:
- Begin with a compelling hook or unexpected anecdote that captures attention
- Weave together multiple narratives and examples that build to larger insights
- Use natural, conversational language suitable for text-to-speech
- Include dramatic pauses and pacing variations (through sentence structure)
- Avoid any formatting like headers, bullet points, or special characters
- Create emotional peaks and valleys to maintain engagement
- End with a powerful conclusion that ties everything together
- Write for the ear, not the eye (no citations, parentheticals, etc.)
- Use short paragraphs and clear transitions between ideas

Write a ~5000 word podcast script that's optimized for text-to-speech delivery."""},
        {"role": "user", "content": f"Title: {title}\n\nCore idea: {idea}\n\nCreate a compelling podcast episode that explores this topic through vivid storytelling and unexpected connections, building to a powerful insight."},
    ]
    resp = chat_create(messages, temperature=0.7, max_tokens=2048)
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    form = """
    <form method='post' action='/generate' class='card p-4 shadow-lg'>
      <div class='mb-3'>
        <label class='form-label' for='idea'>Episode idea / talking points</label>
        <textarea class='form-control' id='idea' name='idea' rows='6' required></textarea>
      </div>
      <button class='btn btn-primary' type='submit' onclick="this.disabled=true; this.innerHTML='Generating... (this may take a minute)'; this.form.submit();">Generate Draft</button>
    </form>"""
    return page(form)

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

    body = f"""
    <form method='post' action='/approve/{draft_id}' class='card shadow-sm p-4'>
      <div class='mb-3'>
        <label class='form-label'>Episode title (editable)</label>
        <input class='form-control' name='title' value='{d['title']}' required />
      </div>
      <div class='mb-3'>
        <label class='form-label'>Script</label>
        <textarea class='form-control' name='script' rows='18' required>{d['script']}</textarea>
      </div>
      <button class='btn btn-success me-2' onclick="this.disabled=true; this.innerHTML='Publishing...'; this.form.submit();">Approve & Publish</button>
      <button formaction='/revise/{draft_id}' formmethod='post' class='btn btn-warning' onclick="this.disabled=true; this.innerHTML='Regenerating...'; this.form.submit();">Regenerate from Feedback</button>
      <div class='mt-3'>
        <textarea class='form-control' name='feedback' placeholder='Optional feedback for regeneration…'></textarea>
      </div>
    </form>"""
    return page(body)

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
