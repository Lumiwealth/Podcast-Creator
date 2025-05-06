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
from concurrent.futures import ThreadPoolExecutor, as_completed
import re # Import regex module
from datetime import datetime # Add datetime import

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

import logging

# Add logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables from .env file
load_dotenv()

# Import helpers from the finished uploader script
import make_episodes_and_upload as uploader

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)
PREFERRED_MODEL = "gpt-4o"   # Latest flagship model (as of May 2024)
FALLBACK_MODEL  = "gpt-4o-mini" # Capable and fast fallback

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
app.template_folder = "templates" # added template folder

# Ephemeral draft storage
DRAFTS: Dict[str, Dict] = {}

# Ensure necessary directories exist at startup
INPUTS_DIR = Path("inputs")
SCRIPTS_DIR = Path("scripts")
AUDIO_DIR = Path("audio")
VIDEO_DIR = Path("video")

for directory in [INPUTS_DIR, SCRIPTS_DIR, AUDIO_DIR, VIDEO_DIR]:
    directory.mkdir(exist_ok=True)

# Create a placeholder in video directory
video_placeholder = VIDEO_DIR / ".placeholder"
if not video_placeholder.exists():
    video_placeholder.touch()
    logging.info(f"Created placeholder file in {VIDEO_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions (New and for File Naming)
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_filename(text: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    text = re.sub(r'[^\w\s-]', '', text) # Remove non-alphanumeric (excluding whitespace, hyphens)
    text = re.sub(r'[-\s]+', '_', text).strip('_') # Replace whitespace/hyphens with underscore
    return text[:100] # Limit length

def get_timestamped_filename_base(title: str) -> str:
    """Generates a base filename: YYYY_MM_DD_SanitizedTitle_ShortUUID"""
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d")
    sanitized_title = sanitize_filename(title)
    short_uuid = uuid.uuid4().hex[:8] # 8-char UUID
    return f"{date_str}_{sanitized_title}_{short_uuid}"

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
    except Exception as e:
        # Log the specific error when the preferred model fails
        logging.warning(f"Preferred model ({PREFERRED_MODEL}) failed: {e}. Falling back to {FALLBACK_MODEL}.")
        # Optionally log more details if available, e.g., e.response for HTTP errors if using httpx client directly
        # or inspect the structure of 'e' if it's an OpenAI specific error class
        try:
            # Attempt to log response body if it's an APIError with a response attribute
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 logging.warning(f"Preferred model error details: {e.response.text}")
        except Exception as log_e:
             logging.warning(f"Could not log detailed error response: {log_e}")

        # Proceed with fallback
        completion = client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            **kw
        )
        return completion

def generate_overall_title(original_prompt: str) -> str:
    """Generates a single, concise, engaging title for the entire episode based on the prompt."""
    logging.info("Generating overall episode title.")
    messages = [
        {"role": "system", "content": "You are a podcast title generator. Create a concise, catchy, and informative title (max 15 words) for a podcast episode based on the following prompt. Output *only* the title itself, without any quotation marks or labels like 'Title:'."},
        {"role": "user", "content": f"Generate a podcast title for this prompt:\n\n{original_prompt}"}
    ]
    try:
        resp = chat_create(messages, temperature=0.7, max_tokens=50)
        title = resp.choices[0].message.content.strip().strip('"') # Remove potential quotes
        logging.info(f"Generated overall title: {title}")
        if not title:
            return "Untitled Episode"
        return title
    except Exception as e:
        logging.exception("Failed to generate overall title.")
        return "Untitled Episode" # Fallback on error

def generate_outline(original_prompt: str, num_parts: int) -> list[str]:
    """Calls AI to break the original prompt into outlines for each part."""
    logging.info(f"Generating outline for {num_parts} parts.")
    if num_parts <= 1:
        return [original_prompt] # No breakdown needed for single part

    prompt = f"""Break the following podcast topic/prompt into a detailed outline for a {num_parts}-part episode series.

For each part, provide a concise summary or list of key points to cover in that specific part. Ensure a logical flow across the parts, with introductions, build-up, and conclusions appropriate for the series structure.

Output *only* the outline for each part, separated by "--- PART BREAK ---". Do not include introductory text before the first part or concluding text after the last part.

Original Topic/Prompt:
{original_prompt}
"""
    messages = [
        {"role": "system", "content": "You are a podcast series planner. You break down a topic into a multi-part episode outline."},
        {"role": "user", "content": prompt}
    ]
    # Use a model good at structuring content, maybe increase max_tokens if outlines are long
    resp = chat_create(messages, temperature=0.5, max_tokens=2048)
    full_outline_text = resp.choices[0].message.content.strip()

    # Split the response into parts
    outlines = [o.strip() for o in full_outline_text.split("--- PART BREAK ---")]

    # Basic validation and cleanup
    if len(outlines) != num_parts:
        logging.warning(f"Expected {num_parts} outlines, but got {len(outlines)}. Using as is, but might be incorrect.")
        # Pad or truncate if necessary, though ideally the model follows instructions
        outlines = (outlines + [''] * num_parts)[:num_parts]
    
    logging.info(f"Generated {len(outlines)} outlines.")
    return outlines


# Define voice tags and corresponding OpenAI voice names
VOICE_PRIMARY_TAG = "[VOICE_1_FEMALE]"
VOICE_SECONDARY_TAG = "[VOICE_2_MALE]"
VOICE_PRIMARY_OPENAI = "alloy"  # Primary female voice
VOICE_SECONDARY_OPENAI = "onyx" # Secondary male voice


def generate_script(part_outline: str, original_prompt: str, part: int, total_parts: int) -> str:
    """Generates the script for a specific part using its outline and the original context."""
    logging.info(f"Generating script for part {part}/{total_parts}")
    part_instruction = ""
    # Refined instructions focusing on content, not part numbers
    if total_parts > 1:
        if part == 1:
            # Focus on introduction and Part 1 content
            part_instruction = f"\n\nAs this is the first part, introduce the overall topic based on the original prompt, then dive *only* into the specific points outlined for Part 1. Do not cover points meant for later parts. End with a compelling transition that hints at the next part's topic without explicitly stating 'next part'."
        elif part == total_parts:
            # Focus on Part N content and conclusion
            part_instruction = f"\n\nThis is the final part. Focus *exclusively* on the specific points outlined for this part. Do not repeat content from previous parts. Conclude the entire episode series, tying back to the original prompt and the journey through the parts."
        else:
            # Focus only on Part X content
            part_instruction = f"\n\nFocus *exclusively* on the specific points outlined for Part {part}. Do not repeat content from previous parts. Ensure a smooth transition from the likely topic of the previous part and end with a compelling transition hinting at the next part's topic without explicitly stating 'next part'."

    system_prompt = f"""You are a master storyteller and expert explainer, writing a podcast script for two voices. Your primary goal is to make complex topics understandable, memorable, and engaging through dialogue. Write a pure narrative script for PART {part} of a {total_parts}-part series.

**Overall Topic Context (from original user prompt):**
{original_prompt}

**Specific Outline for THIS Part ({part}/{total_parts}):**
{part_outline}

**CRITICAL Instructions for Two-Voice Script:**
- **Assign Lines to Voices:** Use two distinct voices:
    - **{VOICE_PRIMARY_TAG}**: The main narrator, a female voice. This voice will carry the primary thread of the content.
    - **{VOICE_SECONDARY_TAG}**: An expert commentator or co-host, a male voice. This voice should provide additional insights, ask clarifying questions, offer different perspectives, or summarize key points.
- **Dynamic Interaction Style:** Infuse a playfully challenging, slightly flirty, and fun dynamic between {VOICE_PRIMARY_TAG} and {VOICE_SECONDARY_TAG}. They should sound like they enjoy working together, occasionally teasing, gently questioning each other's points before building on them, or adding a witty remark. This tension should be light-hearted and serve to make the explanations more engaging, not to create actual conflict or derail the topic. For example, {VOICE_SECONDARY_TAG} might say, "Are you sure about that, {VOICE_PRIMARY_TAG}? I thought it was more like..." and {VOICE_PRIMARY_TAG} could respond with a confident clarification that also acknowledges the point, or vice-versa.
- **Natural Dialogue:** Create a natural, flowing dialogue. The interaction should enhance understanding and engagement. It doesn't need to be strictly alternating lines for the entire script, but intersperse dialogue segments throughout the part, especially when explaining complex topics or transitioning between ideas.
- **Clear Delineation:** Start each line or paragraph with the appropriate tag (e.g., `{VOICE_PRIMARY_TAG} Today, we'll explore...` or `{VOICE_SECONDARY_TAG} That's a great point, and I'd add...`).
- **MANDATORY Structured Explanations:** For any point in this part's outline that involves explaining a concept, process, data, or any new information, YOU ARE REQUIRED to use structured teaching techniques (numbered lists, lettered lists, mnemonics, step-by-step analogies), delivered by either voice or in dialogue. This is non-negotiable.
- **Substantive and Insightful Content:** Deliver in-depth, insightful content. Minimize "fluff."
- **Focus EXCLUSIVELY on THIS Part's Outline.** Do NOT discuss topics assigned to other parts.
- **DO NOT REPEAT content from other parts.**
- **DO NOT explicitly mention the part number.**
- Write the script *only* for Part {part}.
- Use natural, conversational language optimized for text-to-speech.
- Write ONLY narrative text - no audio cues, music notes, pauses, or sound effects.
- Write in an engaging, authoritative, and exceptionally clear style.
{part_instruction}

Write a pure narrative script for Part {part} using two voices with the specified dynamic. Ensure it fits logically within the series. Your main task is to explain the outlined points with structured clarity and depth, using an engaging and fun dialogue.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate the podcast script for Part {part}."}
    ]
    resp = chat_create(messages, temperature=0.7, max_tokens=4096) # Consider adjusting max_tokens based on desired part length
    script = resp.choices[0].message.content.strip()
    logging.info(f"Script generated for part {part}/{total_parts}")
    return script

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    logging.info("Rendering home page")
    return render_template("home.html")

@app.route("/generate", methods=["POST"])
def generate():
    logging.info("Received request to /generate")
    original_prompt = request.form["idea"].strip() # Single prompt
    num_parts = int(request.form.get("num_parts", 1))
    logging.info(f"Original prompt: {original_prompt}")
    logging.info(f"Number of parts: {num_parts}")

    if not original_prompt:
        flash("Please enter a topic or prompt.")
        return redirect(url_for("home"))

    try:
        part_outlines = generate_outline(original_prompt, num_parts)
        overall_title = generate_overall_title(original_prompt)
        
        # Generate base filename for all related files
        base_filename = get_timestamped_filename_base(overall_title)
        
        # Save original prompt
        input_filepath = INPUTS_DIR / f"{base_filename}.txt"
        with open(input_filepath, "w", encoding="utf-8") as f:
            f.write(original_prompt)
        logging.info(f"Original prompt saved to: {input_filepath}")

    except Exception as e:
        logging.exception("Failed during initial generation (outline, title, or input saving).")
        flash(f"Error during initial generation: {e}")
        return redirect(url_for("home"))

    scripts_data = [None] * num_parts # Pre-allocate list
    
    with ThreadPoolExecutor(max_workers=num_parts) as executor:
        future_to_part = {
            executor.submit(generate_script, part_outlines[part-1], original_prompt, part, num_parts): part
            for part in range(1, num_parts + 1)
        }
        logging.info(f"Submitted {num_parts} script generation tasks.")
        for future in as_completed(future_to_part):
            part = future_to_part[future]
            try:
                script = future.result()
                # No title extraction needed here anymore
                scripts_data[part-1] = {"script": script} # Store only script
                logging.info(f"Completed script generation for part {part}/{num_parts}")
            except Exception as exc:
                logging.exception(f"Part {part} script generation failed: {exc}")
                scripts_data[part-1] = None # Mark as failed

    # Combine scripts
    combined_script = ""
    for part_idx, data in enumerate(scripts_data):
        if data:
            if combined_script:
                 combined_script += "\n\n" # Add simple newline separation
            combined_script += data['script']
        else:
             logging.error(f"Part {part_idx+1} script data is missing due to generation failure.")

    if not combined_script.strip():
        flash("Failed to generate any script content.")
        return redirect(url_for("home"))

    # Save combined script
    script_filepath = SCRIPTS_DIR / f"{base_filename}.txt"
    with open(script_filepath, "w", encoding="utf-8") as f:
        f.write(combined_script.strip())
    logging.info(f"Combined script saved to: {script_filepath}")

    draft_id = str(uuid.uuid4())
    DRAFTS[draft_id] = {
        "idea": original_prompt,
        "title": overall_title,
        "script": combined_script.strip(),
        "part": 1, 
        "total_parts": 1,
        "base_filename": base_filename, # Store base filename for later use (MP3)
        "script_filepath": str(script_filepath) # Store script filepath
    }
    logging.info(f"Combined draft created with ID: {draft_id}, base filename: {base_filename}")

    logging.info(f"Redirecting to review page for combined draft: {draft_id}")
    return redirect(url_for("review", draft_id=draft_id))


@app.route("/generate_and_publish", methods=["POST"])
def generate_and_publish():
    logging.info("Received request to /generate_and_publish")
    original_prompt = request.form["idea"].strip()
    num_parts = int(request.form.get("num_parts", 1)) # This is for the initial script generation
    results = []

    if not original_prompt:
        flash("Please enter a topic or prompt.")
        return redirect(url_for("home"))

    # Initial script generation (this part remains largely the same)
    try:
        part_outlines = generate_outline(original_prompt, num_parts)
        overall_title = generate_overall_title(original_prompt)
        base_filename = get_timestamped_filename_base(overall_title)
        input_filepath = INPUTS_DIR / f"{base_filename}.txt"
        with open(input_filepath, "w", encoding="utf-8") as f:
            f.write(original_prompt)
        logging.info(f"Original prompt saved to: {input_filepath}")
    except Exception as e:
        logging.exception("Failed during initial generation (outline, title, or input saving) for publish.")
        flash(f"Error during initial generation: {e}")
        return redirect(url_for("home"))

    # Script generation for each part
    scripts_data_list = [None] * num_parts # Changed variable name to avoid conflict
    with ThreadPoolExecutor(max_workers=num_parts) as executor:
        future_to_part = {
            executor.submit(generate_script, part_outlines[part_idx], original_prompt, part_idx + 1, num_parts): part_idx + 1
            for part_idx in range(num_parts)
        }
        logging.info(f"Submitted {num_parts} script generation tasks for generate_and_publish.")
        for future in as_completed(future_to_part):
            part_num = future_to_part[future]
            try:
                script_part_content = future.result()
                scripts_data_list[part_num-1] = {"script": script_part_content}
                logging.info(f"Completed script generation for part {part_num}/{num_parts} for generate_and_publish.")
            except Exception as exc:
                logging.exception(f"Part {part_num} script generation failed for generate_and_publish: {exc}")
                scripts_data_list[part_num-1] = None
    
    combined_script_content = ""
    successful_parts = 0
    for part_idx, data in enumerate(scripts_data_list):
        if data and data['script']:
            successful_parts += 1
            if combined_script_content:
                 combined_script_content += "\n\n"
            combined_script_content += data['script']
        else:
             logging.error(f"Part {part_idx+1} script data is missing for generate_and_publish.")
    
    combined_script_content = combined_script_content.strip()

    if successful_parts == 0:
        logging.error("All script parts failed generation. Cannot publish.")
        flash("Failed to generate any script content for publishing.")
        return redirect(url_for("home"))
    elif successful_parts < num_parts:
         logging.warning(f"Publishing incomplete episode: {successful_parts}/{num_parts} parts succeeded.")
         results.append(f"⚠️ Published incomplete episode ({successful_parts}/{num_parts} parts)")

    script_filepath = SCRIPTS_DIR / f"{base_filename}.txt"
    with open(script_filepath, "w", encoding="utf-8") as f:
        f.write(combined_script_content)
    logging.info(f"Combined script for publishing saved to: {script_filepath}")

    # TTS and Upload for the combined script
    try:
        logging.info(f"Starting TTS and publish for combined episode: {overall_title}")
        
        # Multi-voice TTS processing for combined_script_content
        raw_segments = re.split(f'({re.escape(VOICE_PRIMARY_TAG)}|{re.escape(VOICE_SECONDARY_TAG)})', combined_script_content)
        parsed_segments_for_tts = []
        current_openai_voice = VOICE_PRIMARY_OPENAI
        idx = 0
        while idx < len(raw_segments):
            segment_text = raw_segments[idx].strip()
            if not segment_text: idx += 1; continue
            if segment_text == VOICE_PRIMARY_TAG:
                current_openai_voice = VOICE_PRIMARY_OPENAI
                if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                    parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
                idx += 2
            elif segment_text == VOICE_SECONDARY_TAG:
                current_openai_voice = VOICE_SECONDARY_OPENAI
                if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                    parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
                idx += 2
            else:
                parsed_segments_for_tts.append((segment_text, current_openai_voice))
                idx += 1

        all_tts_tasks_with_voice = []
        for text_segment, voice_name in parsed_segments_for_tts:
            segment_chunks = uploader.chunk_text(text_segment, uploader.CHUNK_SIZE)
            for chunk_for_tts in segment_chunks:
                all_tts_tasks_with_voice.append((chunk_for_tts, voice_name))

        audio_parts_ordered = [None] * len(all_tts_tasks_with_voice)
        all_tts_succeeded_publish = True

        with ThreadPoolExecutor(max_workers=uploader.MAX_WORKERS) as tts_executor:
            future_to_task_idx = {
                tts_executor.submit(uploader.tts_chunk, f"pub_task_{i}", task_text, task_voice): i
                for i, (task_text, task_voice) in enumerate(all_tts_tasks_with_voice)
            }
            for tts_future in as_completed(future_to_task_idx):
                task_idx = future_to_task_idx[tts_future]
                try:
                    _, audio_content = tts_future.result()
                    audio_parts_ordered[task_idx] = audio_content
                except Exception as tts_exc:
                    logging.exception(f"TTS chunk for pub_task_{task_idx} failed: {tts_exc}")
                    all_tts_succeeded_publish = False
                    # Potentially break here if one failure means we shouldn't publish
                    # For now, it will try to complete others and publish partial if some fail

        if not all_tts_succeeded_publish:
            results.append("⚠️ Some audio parts failed to generate; episode may be incomplete or silent.")
        
        final_audio_segments_publish = [ap for ap in audio_parts_ordered if ap is not None]
        if not final_audio_segments_publish:
             raise ValueError("All TTS chunks failed to generate for publishing.")

        mp3_filename_with_ext = f"{base_filename}.mp3"
        mp3_path = AUDIO_DIR / mp3_filename_with_ext
        with mp3_path.open("wb") as f:
            for part_audio in final_audio_segments_publish:
                f.write(part_audio)
        logging.info(f"MP3 file written for publishing: {mp3_path}")

        up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
        uploader.transistor_put_audio(up_url, mp3_path)
        episode_description = uploader.generate_description(overall_title + ": " + original_prompt[:300]) # Using original prompt for description here
        ep_id = uploader.transistor_create_episode(overall_title, episode_description, audio_url)
        uploader.transistor_publish_episode(ep_id)
        results.append(f"✅ Published “{overall_title}” (ID: {ep_id})")
        logging.info(f"Successfully published episode: {overall_title} (ID: {ep_id})")

    except Exception as e:
        logging.exception(f"Failed to TTS/publish combined episode: {overall_title}")
        results.append(f"❌ {overall_title[:40]}… – {e}")

    flash("<br>".join(results) + '<br><a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">View on Transistor</a>')
    logging.info("Redirecting to home after generate_and_publish")
    return redirect(url_for("home"))

@app.route("/review/<draft_id>")
def review(draft_id):
    logging.info(f"Rendering review page for draft_id: {draft_id}")
    d = DRAFTS.get(draft_id)
    if not d:
        logging.warning(f"Draft not found: {draft_id}")
        flash("Draft not found")
        return redirect(url_for("home"))
    return render_template("review.html", draft_id=draft_id, title=d['title'], script=d['script'], draft=d)

@app.route("/audio/<filename>")
def serve_audio(filename):
    logging.info(f"Serving audio file: {filename}")
    return send_from_directory("audio", filename)

@app.route("/revise/<draft_id>", methods=["POST"])
def revise(draft_id):
    logging.info(f"Received request to revise draft_id: {draft_id}")
    d = DRAFTS.get(draft_id)
    if not d:
        logging.warning(f"Draft not found: {draft_id}")
        flash("Draft not found")
        return redirect(url_for("home"))

    d["title"]  = request.form.get("title", d["title"]).strip()
    new_script_content = request.form.get("script", d["script"]).strip()
    feedback     = request.form.get("feedback", "").strip()

    if feedback:
        logging.info("Regenerating script with feedback")
        messages = [
            {"role": "system", "content": "Improve the podcast script per feedback (~1200 words)."},
            {"role": "assistant", "content": d["script"]}, # Use current script in draft for context
            {"role": "user", "content": feedback},
        ]
        resp = chat_create(messages, temperature=0.7, max_tokens=2048)
        new_script_content = resp.choices[0].message.content.strip()
        logging.info("Script regenerated successfully based on feedback.")
    
    d["script"] = new_script_content # Update script in draft

    # Overwrite the script file with the new content
    if "base_filename" in d:
        script_filepath = SCRIPTS_DIR / f"{d['base_filename']}.txt"
        try:
            with open(script_filepath, "w", encoding="utf-8") as f:
                f.write(d["script"])
            logging.info(f"Revised script saved to: {script_filepath}")
            d["script_filepath"] = str(script_filepath) # Update path in draft
        except Exception as e:
            logging.exception(f"Failed to save revised script to {script_filepath}")
            flash("Error saving revised script file.")
    else:
        logging.warning(f"Cannot save revised script for draft {draft_id}, base_filename missing.")
        flash("Warning: Could not save revised script to file (missing base filename).")
    
    if feedback:
        flash("Script regenerated with feedback and saved.")
    else:
        flash("Script updated and saved.")
    return redirect(url_for("review", draft_id=draft_id))


@app.route("/generate_mp3/<draft_id>", methods=["POST"])
def generate_mp3(draft_id):
    logging.info(f"Received request to generate MP3 for combined draft_id: {draft_id}")
    d = DRAFTS.get(draft_id)
    if not d:
        logging.warning(f"Draft not found: {draft_id}")
        flash("Draft not found")
        return redirect(url_for("home"))

    title = request.form.get("title", d["title"]).strip()
    script_content = request.form.get("script", d["script"]).strip()
    
    # Update draft title and script, and save script file if changed
    if d["title"] != title or d["script"] != script_content:
        d["title"] = title
        d["script"] = script_content
        if "base_filename" in d:
            script_filepath = SCRIPTS_DIR / f"{d['base_filename']}.txt"
            try:
                with open(script_filepath, "w", encoding="utf-8") as f:
                    f.write(script_content)
                logging.info(f"Script updated and saved to: {script_filepath} before MP3 generation.")
                d["script_filepath"] = str(script_filepath)
            except Exception as e:
                logging.exception(f"Failed to save updated script to {script_filepath}")
                flash("Error saving updated script file before MP3 generation.")
        else:
            logging.warning(f"Cannot save updated script for draft {draft_id}, base_filename missing.")

    logging.info("Parsing script for multi-voice TTS")
    raw_segments = re.split(f'({re.escape(VOICE_PRIMARY_TAG)}|{re.escape(VOICE_SECONDARY_TAG)})', script_content)
    
    parsed_segments_for_tts = [] # List of (text, voice_openai_name)
    current_openai_voice = VOICE_PRIMARY_OPENAI # Default
    idx = 0
    while idx < len(raw_segments):
        segment_text = raw_segments[idx].strip()
        if not segment_text:
            idx += 1
            continue

        if segment_text == VOICE_PRIMARY_TAG:
            current_openai_voice = VOICE_PRIMARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        elif segment_text == VOICE_SECONDARY_TAG:
            current_openai_voice = VOICE_SECONDARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        else: # Text not preceded by a known tag, or at the beginning
            parsed_segments_for_tts.append((segment_text, current_openai_voice))
            idx += 1
            
    all_tts_tasks_with_voice = []
    for text_segment, voice_name in parsed_segments_for_tts:
        segment_chunks = uploader.chunk_text(text_segment, uploader.CHUNK_SIZE)
        for chunk_for_tts in segment_chunks:
            all_tts_tasks_with_voice.append((chunk_for_tts, voice_name))

    audio_parts_ordered = [None] * len(all_tts_tasks_with_voice)
    all_succeeded = True

    with ThreadPoolExecutor(max_workers=uploader.MAX_WORKERS) as executor:
        future_to_task_idx = {
            executor.submit(uploader.tts_chunk, f"taskidx_{i}", task_text, task_voice): i
            for i, (task_text, task_voice) in enumerate(all_tts_tasks_with_voice)
        }
        for future in as_completed(future_to_task_idx):
            task_idx = future_to_task_idx[future]
            try:
                _, audio_content = future.result()
                audio_parts_ordered[task_idx] = audio_content
                logging.info(f"TTS chunk for task_idx {task_idx} (voice: {all_tts_tasks_with_voice[task_idx][1]}) completed.")
            except Exception as exc:
                 logging.exception(f"TTS chunk for task_idx {task_idx} failed: {exc}")
                 flash(f"Error generating audio for part of script. MP3 may be incomplete.")
                 audio_parts_ordered[task_idx] = None
                 all_succeeded = False
    
    final_audio_segments = [ap for ap in audio_parts_ordered if ap is not None]

    if not all_succeeded or not final_audio_segments:
         flash("One or more audio chunks failed. MP3 generation may be incomplete or failed.")
         # Decide if to proceed with partial audio or redirect. For now, proceed.

    base_filename = d.get("base_filename")
    if not base_filename:
        logging.error(f"Base filename missing for draft {draft_id}, cannot name MP3 correctly.")
        flash("Critical error: Cannot determine MP3 filename. MP3 not saved.")
        return redirect(url_for("review", draft_id=draft_id))

    mp3_filename_with_ext = f"{base_filename}.mp3"
    mp3_path = AUDIO_DIR / mp3_filename_with_ext

    logging.info(f"Writing {len(final_audio_segments)} audio segments to MP3 file: {mp3_path}")
    with mp3_path.open("wb") as f:
        for part_audio in final_audio_segments:
            f.write(part_audio)

    d["mp3_path"] = str(mp3_path)
    d["mp3_filename"] = mp3_path.name
    logging.info(f"MP3 generated and stored at {mp3_path}")
    flash_message = "MP3 generated successfully!"
    if not all_succeeded:
        flash_message += " (Some parts may be missing due to errors)"
    flash(flash_message)
    return redirect(url_for("review", draft_id=draft_id, mp3_ready=True))


@app.route("/upload/<draft_id>", methods=["POST"])
def upload(draft_id):
    logging.info(f"Received request to upload combined draft_id: {draft_id}")
    d = DRAFTS.get(draft_id)
    if not d or "mp3_path" not in d:
        logging.warning(f"MP3 not found for draft_id: {draft_id}")
        flash("MP3 not found or draft missing")
        return redirect(url_for("review", draft_id=draft_id) if d else url_for("home"))

    title = request.form.get("title", d["title"]).strip()
    script = request.form.get("script", d["script"]).strip() # Full script for description
    mp3_path = Path(d["mp3_path"]) # mp3_path should now use base_filename.mp3

    # Update draft title and script, and save script file if changed by user on review page
    if d["title"] != title or d["script"] != script:
        d["title"] = title
        d["script"] = script
        if "base_filename" in d:
            script_filepath = SCRIPTS_DIR / f"{d['base_filename']}.txt"
            try:
                with open(script_filepath, "w", encoding="utf-8") as f:
                    f.write(script)
                logging.info(f"Script updated and saved to: {script_filepath} before upload.")
                d["script_filepath"] = str(script_filepath)
            except Exception as e:
                logging.exception(f"Failed to save updated script to {script_filepath}")
                flash("Error saving updated script file before upload.")
        else:
            logging.warning(f"Cannot save updated script for draft {draft_id}, base_filename missing.")


    try:
        logging.info("Authorizing upload to Transistor")
        # mp3_path.name will be base_filename.mp3
        up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
        logging.info("Uploading MP3 to Transistor S3")
        uploader.transistor_put_audio(up_url, mp3_path)
        logging.info("Creating episode draft on Transistor")
        # Generate description from overall title and combined script
        episode_description = uploader.generate_description(title + ": " + script[:300]) # title is overall_title from draft
        ep_id = uploader.transistor_create_episode(title, episode_description, audio_url)

        logging.info(f"Draft uploaded to Transistor (ID {ep_id})")
        flash(f'Episode drafted on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">Publish your episode here</a>')
        # Remove draft after successful upload? Or keep until published? Let's keep for now.
        # DRAFTS.pop(draft_id, None)
        return redirect(url_for("home"))
    except Exception as e:
        logging.exception(f"Failed to upload draft {draft_id}: {e}")
        flash(f"Error uploading to Transistor: {e}")
        return redirect(url_for("review", draft_id=draft_id))


@app.route("/generate_and_upload/<draft_id>", methods=["POST"])
def generate_and_upload(draft_id):
    logging.info(f"Received request to generate and upload combined draft_id: {draft_id}")
    d = DRAFTS.get(draft_id)
    if not d:
        logging.warning(f"Draft not found: {draft_id}")
        flash("Draft not found")
        return redirect(url_for("home"))

    title = request.form.get("title", d["title"]).strip()
    script_content = request.form.get("script", d["script"]).strip()

    if d["title"] != title or d["script"] != script_content:
        d["title"] = title
        d["script"] = script_content
        if "base_filename" in d:
            script_filepath = SCRIPTS_DIR / f"{d['base_filename']}.txt"
            try:
                with open(script_filepath, "w", encoding="utf-8") as f:
                    f.write(script_content)
                logging.info(f"Script updated and saved to: {script_filepath} before TTS & upload.")
                d["script_filepath"] = str(script_filepath)
            except Exception as e:
                logging.exception(f"Failed to save updated script to {script_filepath}")
                flash("Error saving updated script file.")
        else:
            logging.warning(f"Cannot save updated script for draft {draft_id}, base_filename missing.")


    logging.info("Parsing script for multi-voice TTS for generate_and_upload")
    raw_segments = re.split(f'({re.escape(VOICE_PRIMARY_TAG)}|{re.escape(VOICE_SECONDARY_TAG)})', script_content)
    parsed_segments_for_tts = []
    current_openai_voice = VOICE_PRIMARY_OPENAI
    idx = 0
    while idx < len(raw_segments):
        segment_text = raw_segments[idx].strip()
        if not segment_text:
            idx += 1
            continue
        if segment_text == VOICE_PRIMARY_TAG:
            current_openai_voice = VOICE_PRIMARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        elif segment_text == VOICE_SECONDARY_TAG:
            current_openai_voice = VOICE_SECONDARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        else:
            parsed_segments_for_tts.append((segment_text, current_openai_voice))
            idx += 1
            
    all_tts_tasks_with_voice = []
    for text_segment, voice_name in parsed_segments_for_tts:
        segment_chunks = uploader.chunk_text(text_segment, uploader.CHUNK_SIZE)
        for chunk_for_tts in segment_chunks:
            all_tts_tasks_with_voice.append((chunk_for_tts, voice_name))

    audio_parts_ordered = [None] * len(all_tts_tasks_with_voice)
    all_succeeded = True
    with ThreadPoolExecutor(max_workers=uploader.MAX_WORKERS) as executor:
        future_to_task_idx = {
            executor.submit(uploader.tts_chunk, f"taskidx_{i}", task_text, task_voice): i
            for i, (task_text, task_voice) in enumerate(all_tts_tasks_with_voice)
        }
        for future in as_completed(future_to_task_idx):
            task_idx = future_to_task_idx[future]
            try:
                _, audio_content = future.result()
                audio_parts_ordered[task_idx] = audio_content
                logging.info(f"TTS chunk for task_idx {task_idx} (voice: {all_tts_tasks_with_voice[task_idx][1]}) completed for generate_and_upload.")
            except Exception as exc:
                 logging.exception(f"TTS chunk for task_idx {task_idx} failed in generate_and_upload: {exc}")
                 flash(f"Error generating audio for part of script. Upload cancelled.")
                 all_succeeded = False
                 break 
    
    if not all_succeeded:
         return redirect(url_for("review", draft_id=draft_id))

    final_audio_segments = [ap for ap in audio_parts_ordered if ap is not None]
    if len(final_audio_segments) != len(all_tts_tasks_with_voice): # Should not happen if all_succeeded is true and we broke on first error
        logging.error("Mismatch in expected audio segments after TTS. Aborting upload.")
        flash("Critical error during audio generation. Upload cancelled.")
        return redirect(url_for("review", draft_id=draft_id))


    base_filename = d.get("base_filename")
    # ... (base_filename check as before) ...
    if not base_filename:
        logging.error(f"Base filename missing for draft {draft_id}, cannot name MP3 correctly for generate_and_upload.")
        flash("Critical error: Cannot determine MP3 filename. Process cancelled.")
        return redirect(url_for("review", draft_id=draft_id))

    mp3_filename_with_ext = f"{base_filename}.mp3"
    mp3_path = AUDIO_DIR / mp3_filename_with_ext

    logging.info(f"Writing {len(final_audio_segments)} audio segments to MP3 file: {mp3_path}")
    with mp3_path.open("wb") as f:
        for part_audio in final_audio_segments:
            f.write(part_audio)

    d["mp3_path"] = str(mp3_path)
    d["mp3_filename"] = mp3_path.name

    try:
        logging.info("Authorizing upload to Transistor")
        up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
        logging.info("Uploading MP3 to Transistor S3")
        uploader.transistor_put_audio(up_url, mp3_path)
        logging.info("Creating episode draft on Transistor")
        # Use full script_content for description, tags will be included but might be fine for short desc.
        episode_description = uploader.generate_description(title + ": " + script_content[:300]) 
        ep_id = uploader.transistor_create_episode(title, episode_description, audio_url)

        logging.info(f"Draft uploaded to Transistor (ID {ep_id})")
        flash(f'✅ Draft uploaded to Transistor (ID {ep_id})')
        d["transistor_id"] = ep_id
        d["uploaded"] = True
        return redirect(url_for("review", draft_id=draft_id, uploaded=1))
    except Exception as e:
        logging.exception(f"Failed to upload draft {draft_id}: {e}")
        flash(f"Error uploading to Transistor: {e}")
        return redirect(url_for("review", draft_id=draft_id, mp3_ready=True))


@app.route("/approve/<draft_id>", methods=["POST"])
def approve(draft_id):
    logging.info(f"Received request to approve combined draft_id: {draft_id}")
    d = DRAFTS.get(draft_id) 
    if not d:
        logging.warning(f"Draft not found: {draft_id}")
        flash("Draft not found")
        return redirect(url_for("home"))

    title  = request.form.get("title", d["title"]).strip()
    script_content = request.form.get("script", d["script"]).strip()

    if d["title"] != title or d["script"] != script_content:
        d["title"] = title
        d["script"] = script_content
        # ... (save script file logic as before) ...
        if "base_filename" in d:
            script_filepath = SCRIPTS_DIR / f"{d['base_filename']}.txt"
            try:
                with open(script_filepath, "w", encoding="utf-8") as f:
                    f.write(script_content)
                logging.info(f"Script updated and saved to: {script_filepath} before approval.")
                d["script_filepath"] = str(script_filepath)
            except Exception as e:
                logging.exception(f"Failed to save updated script to {script_filepath}")
                flash("Error saving updated script file before approval.")
        else:
            logging.warning(f"Cannot save updated script for draft {draft_id}, base_filename missing.")


    logging.info("Parsing script for multi-voice TTS for approval")
    raw_segments = re.split(f'({re.escape(VOICE_PRIMARY_TAG)}|{re.escape(VOICE_SECONDARY_TAG)})', script_content)
    parsed_segments_for_tts = []
    current_openai_voice = VOICE_PRIMARY_OPENAI
    idx = 0
    while idx < len(raw_segments):
        segment_text = raw_segments[idx].strip()
        if not segment_text:
            idx += 1
            continue
        if segment_text == VOICE_PRIMARY_TAG:
            current_openai_voice = VOICE_PRIMARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        elif segment_text == VOICE_SECONDARY_TAG:
            current_openai_voice = VOICE_SECONDARY_OPENAI
            if idx + 1 < len(raw_segments) and raw_segments[idx+1].strip():
                parsed_segments_for_tts.append((raw_segments[idx+1].strip(), current_openai_voice))
            idx += 2
        else:
            parsed_segments_for_tts.append((segment_text, current_openai_voice))
            idx += 1
            
    all_tts_tasks_with_voice = []
    for text_segment, voice_name in parsed_segments_for_tts:
        segment_chunks = uploader.chunk_text(text_segment, uploader.CHUNK_SIZE)
        for chunk_for_tts in segment_chunks:
            all_tts_tasks_with_voice.append((chunk_for_tts, voice_name))

    audio_parts_ordered = [None] * len(all_tts_tasks_with_voice)
    all_succeeded = True
    with ThreadPoolExecutor(max_workers=uploader.MAX_WORKERS) as executor:
        future_to_task_idx = {
            executor.submit(uploader.tts_chunk, f"taskidx_{i}", task_text, task_voice): i
            for i, (task_text, task_voice) in enumerate(all_tts_tasks_with_voice)
        }
        for future in as_completed(future_to_task_idx):
            task_idx = future_to_task_idx[future]
            try:
                _, audio_content = future.result()
                audio_parts_ordered[task_idx] = audio_content
                logging.info(f"TTS chunk for task_idx {task_idx} (voice: {all_tts_tasks_with_voice[task_idx][1]}) completed for approve.")
            except Exception as exc:
                 logging.exception(f"TTS chunk for task_idx {task_idx} failed in approve: {exc}")
                 flash(f"Error generating audio for part of script. Upload cancelled.")
                 all_succeeded = False
                 break 

    if not all_succeeded:
         return redirect(url_for("review", draft_id=draft_id))

    final_audio_segments = [ap for ap in audio_parts_ordered if ap is not None]
    if len(final_audio_segments) != len(all_tts_tasks_with_voice):
        logging.error("Mismatch in expected audio segments after TTS for approval. Aborting.")
        flash("Critical error during audio generation for approval. Process cancelled.")
        return redirect(url_for("review", draft_id=draft_id))

    base_filename = d.get("base_filename")
    # ... (base_filename check as before) ...
    if not base_filename:
        logging.error(f"Base filename missing for draft {draft_id}, cannot name MP3 for approval.")
        flash("Critical error: Cannot determine MP3 filename for approval. Process cancelled.")
        return redirect(url_for("review", draft_id=draft_id))

    mp3_filename_with_ext = f"{base_filename}.mp3"
    mp3_path = AUDIO_DIR / mp3_filename_with_ext

    logging.info(f"Writing {len(final_audio_segments)} audio segments to MP3 file: {mp3_path}")
    with mp3_path.open("wb") as f:
        for part_audio in final_audio_segments:
            f.write(part_audio)

    # Upload to Transistor (draft)
    try:
        logging.info("Authorizing upload to Transistor")
        # mp3_path.name will be base_filename.mp3
        up_url, audio_url = uploader.transistor_authorise(mp3_path.name)
        logging.info("Uploading MP3 to Transistor S3")
        uploader.transistor_put_audio(up_url, mp3_path)
        logging.info("Creating episode draft on Transistor")
        episode_description = uploader.generate_description(title + ": " + script_content[:300])
        ep_id = uploader.transistor_create_episode(title, episode_description, audio_url)

        # Decide if "Approve" should also publish. Currently it doesn't.
        # uploader.transistor_publish_episode(ep_id)

        logging.info(f"Draft uploaded to Transistor (ID {ep_id})")
        flash(f'Episode drafted on Transistor (ID {ep_id}). <a href="https://dashboard.transistor.fm/shows/private-for-lumiwealth-podcast" target="_blank">Publish your episode here</a>')
        # Remove draft from memory ONLY after successful upload
        DRAFTS.pop(draft_id, None)
        logging.info(f"Draft {draft_id} (base: {base_filename}) successfully processed and removed from memory.")
        return redirect(url_for("home"))
    except Exception as e:
        logging.exception(f"Failed to upload approved draft {draft_id}: {e}")
        flash(f"Error uploading approved draft to Transistor: {e}")
        d["mp3_path"] = str(mp3_path) 
        d["mp3_filename"] = mp3_path.name
        return redirect(url_for("review", draft_id=draft_id, mp3_ready=True))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("Starting Flask app")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)