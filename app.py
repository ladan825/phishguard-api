from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
import gdown

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://phishguard-gamma-ten.vercel.app",
])

# Raised from 0.65 -> 0.70 based on diagnostics: real false positives
# ("submit your credentials/timesheet on Monday/Friday") scored 0.615-0.686,
# while a genuine phishing example scored 0.810. 0.70 clears both false
# positives while still catching real phishing. Re-tune against a full
# labeled test set (precision/recall sweep) when you have one.
THRESHOLD = 0.70

# ── Download models ────────────────────────────────────────
def download_models():
    if not os.path.exists('phishing_model.pkl'):
        print("Downloading model...")
        gdown.download(
            'https://drive.google.com/uc?id=1StF2zFfEEFyNWGMvR0IdTYcVmYcbzDcZ',
            'phishing_model.pkl', quiet=False
        )
    if not os.path.exists('vectorizer.pkl'):
        print("Downloading vectorizer...")
        gdown.download(
            'https://drive.google.com/uc?id=1xh2cKcALIgmS6mvkDPYCWjxR9qAl4uUs',
            'vectorizer.pkl', quiet=False
        )

download_models()

# ── Load ───────────────────────────────────────────────────
ensemble_model = joblib.load('phishing_model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')
print("Model loaded!")

# ── Database ───────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            keywords TEXT,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(email_text, result, confidence, keywords):
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (email_text, result, confidence, keywords, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        email_text[:500],
        result,
        confidence,
        ', '.join(keywords),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect('phishguard.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='PHISHING'")
    phishing = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE result='SAFE'")
    safe = c.fetchone()[0]
    conn.close()
    return {'total': total, 'phishing': phishing, 'safe': safe}

# ── Keywords ───────────────────────────────────────────────
def get_keywords(vec):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = vec.toarray()[0]
    rf_model = ensemble_model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    combined_scores = tfidf_scores * importances
    nonzero = np.where(tfidf_scores > 0)[0]
    if len(nonzero) == 0:
        return []
    top_indices = nonzero[np.argsort(combined_scores[nonzero])[::-1]][:5]
    return [feature_names[i] for i in top_indices]

# ── Routes ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    vec = tfidf_vectorizer.transform([text])
    prob_phishing = ensemble_model.predict_proba(vec)[0][1]
    is_phishing = prob_phishing >= THRESHOLD
    confidence = round(float(prob_phishing if is_phishing else 1 - prob_phishing) * 100, 2)
    keywords = get_keywords(vec)
    result = 'PHISHING' if is_phishing else 'SAFE'

    save_prediction(text, result, confidence, keywords)

    return jsonify({
        'result': result,
        'confidence': confidence,
        'keywords': keywords,
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
        # NOTE: email_text is now included -- it was missing before, which
        # meant history items loaded with no text to display.
        c.execute("SELECT id, email_text, result, confidence, keywords, timestamp FROM predictions ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        return jsonify([{
            'id': r[0],
            'text': r[1],
            'result': r[2],
            'confidence': r[3],
            # keywords is stored as a comma-joined string in the DB;
            # split it back into a list so the frontend's .map() works.
            'keywords': [k.strip() for k in r[4].split(',')] if r[4] else [],
            'timestamp': r[5]
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'version': '4.0'
    })

# ── Start ──────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

init_db()