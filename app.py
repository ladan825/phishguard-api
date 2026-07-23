from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
import gdown
import shap
import anthropic

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://phishguard-gamma-ten.vercel.app",
])

THRESHOLD = 0.70
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Download models ────────────────────────────────────────
def download_models():
    if not os.path.exists('phishing_model.pkl'):
        print("Downloading model...")
        gdown.download('https://drive.google.com/uc?id=1StF2zFfEEFyNWGMvR0IdTYcVmYcbzDcZ', 'phishing_model.pkl', quiet=False)
    if not os.path.exists('vectorizer.pkl'):
        print("Downloading vectorizer...")
        gdown.download('https://drive.google.com/uc?id=1xh2cKcALIgmS6mvkDPYCWjxR9qAl4uUs', 'vectorizer.pkl', quiet=False)

download_models()
ensemble_model = joblib.load('phishing_model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')
print("Model loaded!")

# ── SHAP explainer (uses RF from ensemble) ─────────────────
rf_model = ensemble_model.named_estimators_['rf']
shap_explainer = shap.TreeExplainer(rf_model)
feature_names = tfidf_vectorizer.get_feature_names_out()
print("SHAP explainer ready!")

# ── Database ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email_text TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence REAL NOT NULL,
        keywords TEXT,
        explanation TEXT,
        timestamp TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def save_prediction(email_text, result, confidence, keywords, explanation=""):
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
        (email_text, result, confidence, keywords, explanation, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (email_text[:500], result, confidence,
         ', '.join(keywords), explanation,
         datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions"); total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='PHISHING'"); phishing = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='SAFE'"); safe = c.fetchone()[0]
    conn.close()
    return {'total': total, 'phishing': phishing, 'safe': safe}

# ── SHAP keyword extraction ────────────────────────────────
def get_shap_keywords(vec, is_phishing):
    tfidf_scores = vec.toarray()[0]
    nonzero = np.where(tfidf_scores > 0)[0]
    if len(nonzero) == 0:
        return []
    shap_values = shap_explainer.shap_values(vec)
    # shap_values[1] = contribution toward phishing class
    phishing_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    if is_phishing:
        # Top words pushing TOWARD phishing
        top_indices = np.argsort(phishing_shap[nonzero])[::-1][:5]
    else:
        # Top words pushing TOWARD safe (negative shap = away from phishing)
        top_indices = np.argsort(phishing_shap[nonzero])[:5]
    return [feature_names[nonzero[i]] for i in top_indices]

# ── NLP Explanation via Anthropic ──────────────────────────
def get_nlp_explanation(email_text, result, confidence, shap_words):
    """Generate plain English explanation using SHAP words — no API needed"""
    if not shap_words:
        if result == 'PHISHING':
            return f"This email was flagged as phishing with {confidence}% confidence based on its overall linguistic pattern, which closely resembles known phishing email content in the training data."
        else:
            return f"This email appears legitimate with {confidence}% confidence. Its content and phrasing are consistent with normal communication patterns found in legitimate emails."

    word_list = ', '.join(f'"{w}"' for w in shap_words[:3])

    if result == 'PHISHING':
        if confidence >= 90:
            return f"This email is highly likely to be a phishing attempt. The terms {word_list} are strongly associated with phishing campaigns in the model's training data, and the overall pattern of language used matches known malicious email templates with {confidence}% confidence."
        elif confidence >= 75:
            return f"This email shows significant phishing indicators. The presence of {word_list} contributed most to this classification, as these terms frequently appear in emails designed to deceive recipients into disclosing sensitive information."
        else:
            return f"This email was flagged as potentially suspicious. The terms {word_list} appear in patterns associated with phishing emails, though the confidence level of {confidence}% suggests some ambiguity — exercise caution before responding or clicking any links."
    else:
        if confidence >= 80:
            return f"This email appears legitimate. The terms {word_list} are characteristic of normal institutional or personal communication and do not match patterns associated with phishing attempts in the training data."
        else:
            return f"This email is likely legitimate but the classification is borderline at {confidence}% confidence. The terms {word_list} suggest genuine communication, however verify the sender's identity before sharing any sensitive information."

# ── Routes ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Short email guard — also catches greetings and casual messages
    word_count = len(text.split())
    SAFE_PATTERNS = ['hi', 'hey', 'hello', 'ok', 'okay', 'thanks', 'thank you',
                     'yes', 'no', 'sure', 'noted', 'alright', 'bye', 'goodbye']
    if word_count < 8 or text.lower().strip() in SAFE_PATTERNS:
        return jsonify({
            'result': 'SAFE',
            'confidence': 50.0,
            'keywords': [],
            'explanation': 'This message is too short to analyse reliably. Please paste the full email content for accurate detection.',
            'probability': 50.0
        })

    vec = tfidf_vectorizer.transform([text])
    prob_phishing = ensemble_model.predict_proba(vec)[0][1]
    is_phishing = prob_phishing >= THRESHOLD
    confidence = round(float(prob_phishing if is_phishing else 1 - prob_phishing) * 100, 2)
    keywords = get_shap_keywords(vec, is_phishing)
    result = 'PHISHING' if is_phishing else 'SAFE'

    # Get NLP explanation
    explanation = get_nlp_explanation(text, result, confidence, keywords)

    save_prediction(text, result, confidence, keywords, explanation or "")

    return jsonify({
        'result': result,
        'confidence': confidence,
        'keywords': keywords,
        'explanation': explanation,
        'probability': round(float(prob_phishing) * 100, 2)
    })

@app.route('/stats', methods=['GET'])
def stats():
    try:
        return jsonify(get_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        conn = sqlite3.connect('phishguard.db')
        c = conn.cursor()
        c.execute("""SELECT id, email_text, result, confidence, keywords, 
                     explanation, timestamp FROM predictions 
                     ORDER BY id DESC LIMIT 20""")
        rows = c.fetchall()
        conn.close()
        return jsonify([{
            'id': r[0],
            'text': r[1],
            'result': r[2],
            'confidence': r[3],
            'keywords': [k.strip() for k in r[4].split(',')] if r[4] else [],
            'explanation': r[5],
            'timestamp': r[6]
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'version': '5.0'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

init_db()