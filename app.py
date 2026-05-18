from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from transformers import pipeline
from dotenv import load_dotenv
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import google.genai as genai
import re
import logging

# --- Basic app/config ---
app = Flask(__name__, static_folder='static', template_folder='templates')
load_dotenv()

# --- API Keys and Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
SENDGRID_FROM_EMAIL = os.getenv('SENDGRID_FROM_EMAIL')

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Initialize classifier (can take time) ---
# Keep as it is; ensure transformers and model are available
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# --- TMDB helper ---
def get_movie_details(title):
    """Search TMDB for best match for title. Returns dict or None."""
    if not TMDB_API_KEY:
        app.logger.warning("TMDB_API_KEY is not set; get_movie_details cannot fetch posters/links.")
        return {'title': title, 'poster': None, 'link': None}

    safe_title = requests.utils.quote(title.strip())
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={safe_title}"
    try:
        response = requests.get(search_url, timeout=8)
    except Exception as e:
        app.logger.error(f"TMDB request failed for '{title}': {e}")
        return None

    if response.status_code != 200:
        app.logger.error(f"TMDB API returned status {response.status_code} for title '{title}'")
        return None

    results = response.json().get('results', [])
    if results:
        movie = results[0]
        poster_path = movie.get('poster_path')
        return {
            'title': movie.get('title') or title,
            'poster': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
            'link': f"https://www.themoviedb.org/movie/{movie.get('id')}" if movie.get('id') else None
        }

    # relaxed fallback: try first 80 chars (some titles are padded with extra text)
    short_title = title.strip()[:80]
    if short_title and short_title != title:
        app.logger.info(f"No direct TMDB result; trying short title fallback for '{short_title}'")
        return get_movie_details(short_title)

    app.logger.info(f"No TMDB result for '{title}'")
    return None

# --- Parsing helper for model output ---
def extract_titles_from_model_text(text):
    """
    Try multiple heuristics to extract 3 titles from the model text:
      1) Lines that look numbered: "1. Title"
      2) Lines with quotes: "Title"
      3) Plain newline-separated short lines
      4) Comma-separated fallback
    Returns a list of cleaned title strings.
    """
    if not text:
        return []

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    titles = []

    # 1) Numbered lines
    numbered = re.findall(r'^\s*\d+\.\s*(.+)$', text, flags=re.MULTILINE)
    if numbered:
        titles = [clean_title(t) for t in numbered if clean_title(t)]
        if titles:
            return titles

    # 2) Quoted strings
    quoted = re.findall(r'["“](.+?)["”]', text)
    if quoted:
        titles = [clean_title(t) for t in quoted if clean_title(t)]
        if titles:
            return titles

    # 3) Newline-separated short lines (filter out long sentences)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    short_lines = [l for l in lines if len(l.split()) <= 8]  # heuristic: titles are short
    if short_lines:
        titles = [clean_title(l) for l in short_lines if clean_title(l)]
        if titles:
            return titles

    # 4) Comma-separated fallback
    parts = [p.strip() for p in re.split(r',|\n', text) if p.strip()]
    if parts:
        titles = [clean_title(p) for p in parts if clean_title(p)]
        if titles:
            return titles

    # No good extraction
    return []

def clean_title(raw):
    """Do light cleaning: remove leading numbering, stray text like 'here are 3 titles:', trailing punctuation."""
    if not raw:
        return None
    s = raw.strip()
    # remove leading numbering or bullets
    s = re.sub(r'^[\d\-\)\.]+\s*', '', s)
    # remove parenthesized commentary if it's long (e.g., '(a heartwarming film)')
    s = re.sub(r'\s*\(.*?\)\s*$', '', s)
    # remove leading phrases that are not titles
    s = re.sub(r'^(title:|movie:)\s*', '', s, flags=re.I)
    # limit length
    s = s.strip(' "\'')
    if len(s) == 0:
        return None
    # If it's clearly a sentence rather than a title (contains verbs), we still return it — TMDB search may fail but that's okay
    return s

# --- Routes ---
@app.route('/')
def index():
    return render_template('mood_check.html')

@app.route('/how_its_work')
def how_its_work():
    return render_template('how_its_work.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

# compatibility routes for older links that request *.html directly
@app.route('/how_its_work.html')
def how_its_work_html():
    return redirect(url_for('how_its_work'))

@app.route('/about_us.html')
def about_us_html():
    return redirect(url_for('about_us'))

@app.route('/help.html')
def help_html():
    return redirect(url_for('help_page'))

@app.route('/mood_check.html')
def mood_check_html():
    return redirect(url_for('index'))

# static image root shortcuts (optional)
@app.route('/faisal1.jpg')
def faisal1_root():
    return redirect(url_for('static', filename='images/faisal1.jpg'))

@app.route('/faisal-image.png')
def faisalimage_root():
    return redirect(url_for('static', filename='images/faisal-image.png'))

@app.route('/kartikey-image.png')
def kartikey_root():
    return redirect(url_for('static', filename='images/kartikey-image.png'))

# --- Analysis endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        user_input = request.form.get('user_input', '').strip()
        user_email = request.form.get('user_email', '').strip() or None

        app.logger.info(f"User input received: {user_input}")

        if not user_input:
            return jsonify({'error': 'Please provide some text describing your mood.'}), 400

        # Emotion detection
        classification = classifier(user_input)[0]
        emotion = classification.get('label', 'unknown')
        app.logger.info(f"Detected emotion: {emotion}")

        # If GEMINI key missing, skip model generation and use a simple mapping fallback
        if not GEMINI_API_KEY:
            app.logger.warning("GEMINI_API_KEY not set; using fallback static movies based on emotion.")
            fallback_mapping = {
                'joy': ['Paddington 2', 'La La Land', 'School of Rock'],
                'sadness': ['The Pursuit of Happyness', 'Inside Out', 'Lost in Translation'],
                'anger': ['Fight Club', 'Uncut Gems', 'Mad Max: Fury Road'],
                'fear': ['Get Out', 'A Quiet Place', 'The Babadook'],
                'surprise': ['The Prestige', 'Inception', 'Memento'],
                'love': ['Before Sunrise', 'Pride & Prejudice', 'The Notebook'],
            }
            titles = fallback_mapping.get(emotion.lower(), ['The Shawshank Redemption', 'Forrest Gump', 'The Grand Budapest Hotel'])
        else:
            # Use Google GenAI client
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_message = f"You are a helpful movie recommender. The user's detected emotion is: {emotion}."
            prompt = (
                f"The user is feeling {emotion}.\n"
                "Provide exactly 3 movie titles (titles only). Return them as a numbered list (1., 2., 3.) or newline-separated, nothing else.\n"
            )
            app.logger.info("Calling Gemini model for movie suggestions (requested: 3 titles).")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "system_instruction": system_message
                }
            )

            # defensive: try to extract text from response
            model_text = ""
            if hasattr(response, 'text') and response.text:
                model_text = response.text
            elif isinstance(response, dict) and response.get('text'):
                model_text = response.get('text')
            else:
                # Some SDKs return an object where content string is in a nested structure; try to str() as last resort
                model_text = str(response)

            app.logger.info(f"Model raw output: {model_text[:400]}")  # log first chunk for debugging
            titles = extract_titles_from_model_text(model_text)

            if not titles:
                app.logger.warning("Failed to parse model output into titles; using comma-split fallback.")
                # last-resort: comma split
                titles = [t.strip() for t in re.split(r',|\n', model_text) if t.strip()][:3]

        # ensure only up to 3 titles
        titles = titles[:3]

        movies = []
        for title in titles:
            movie_details = get_movie_details(title)
            if movie_details:
                movies.append(movie_details)
            else:
                app.logger.warning(f"Could not find details for movie: {title}")

        result = {'emotion': emotion, 'movies': movies}

        if user_email and movies and SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
            email_sent = send_recommendation_email(user_email, emotion, movies)
            result['email_sent'] = email_sent

        return jsonify(result)

    except Exception as e:
        app.logger.exception("An unexpected error occurred in analyze:")
        error_message = str(e)
        if "API key" in error_message or "quota" in error_message or "429" in error_message:
            error_message = "API Error: Please check your GEMINI_API_KEY and ensure your account has sufficient quota."
        return jsonify({'error': error_message}), 500
#
'''def send_recommendation_email(to_email, emotion, movies):
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        app.logger.error("SendGrid config missing; cannot send email.")
        return False

    message = Mail(
        from_email=SENDGRID_FROM_EMAIL,
        to_emails=to_email,
        subject='Your Movie Recommendations Based on Your Mood',
        html_content=f'''
#            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
#                <h1>🎬 MovieFindr Recommendations</h1>
#                <p>Detected mood: <strong>{emotion}</strong></p>
#                <ul>
#                {''.join([f"<li><a href='{(movie['link'] or '#')}' #target='_blank'>{movie['title']}</a></li>" for movie in movies])}
 #               </ul>
 #           </div>
 #       '''
#    )
#   try:
    #    sg = SendGridAPIClient(SENDGRID_API_KEY)
     #   response = sg.send(message)
      #  app.logger.info(f"Email sent. Status code: {response.status_code}")
      #  return True
   # except Exception as e:
#        app.logger.error(f"Error sending email: {str(e)}")
#        return False
#
def send_recommendation_email(to_email, emotion, movies):
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        app.logger.error("SendGrid config missing; cannot send email.")
        return False

    message = Mail(
        from_email=SENDGRID_FROM_EMAIL,
        to_emails=to_email,
        subject='Your Movie Recommendations Based on Your Mood',
        html_content=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
                <h1>🎬 MovieFindr Recommendations</h1>
                <p>Detected mood: <strong>{emotion}</strong></p>
                <ul>
                {''.join([f"<li><a href='{(movie['link'] or '#')}' target='_blank'>{movie['title']}</a></li>" for movie in movies])}
                </ul>
            </div>
        '''
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        app.logger.info(f"Email sent. Status code: {response.status_code}")
        # Log response body when not 2xx for debugging
        if response.status_code >= 400:
            try:
                body = response.body.decode('utf-8') if hasattr(response, 'body') else str(response.body)
            except Exception:
                body = str(response.body)
            app.logger.error(f"SendGrid returned error. Status: {response.status_code}, Body: {body}")
            return False
        return True
    except Exception as e:
        # Try to extract HTTP response details when using SendGrid's HTTPError
        app.logger.exception("Error sending email:")
        # If the exception has a .status_code or .body, log them
        try:
            status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            body = getattr(e, 'body', None) or getattr(e, 'message', None)
            app.logger.error(f"SendGrid exception details - status: {status}, body: {body}")
        except Exception:
            pass
        return False


if __name__ == '__main__':
    # show Werkzeug logs
    app.run(debug=True, host='127.0.0.1', port=5000)
