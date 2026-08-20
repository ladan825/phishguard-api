from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
import re
import gdown

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://phishguard-gamma-ten.vercel.app",
])

THRESHOLD = 0.65

def download_models():
    if not os.path.exists('phishing_model.pkl'):
        print("Downloading model...")
        gdown.download('https://drive.google.com/uc?id=12-5oQOOYI2C9CL4eodWa2q3zrqW1KSIh', 'phishing_model.pkl', quiet=False)
    if not os.path.exists('vectorizer.pkl'):
        print("Downloading vectorizer...")
        gdown.download('https://drive.google.com/uc?id=1R_1ke4Z5za379zX5yyHBtr2UOBCE2Fst', 'vectorizer.pkl', quiet=False)

download_models()
ensemble_model = joblib.load('phishing_model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
rf_model = ensemble_model.named_estimators_['rf']
lr_model = ensemble_model.named_estimators_['lr']
rf_importances = rf_model.feature_importances_
lr_coef = lr_model.coef_[0]
print("Model loaded!")

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

def get_keywords(vec, is_phishing):
    tfidf_scores = vec.toarray()[0]
    nonzero = np.where(tfidf_scores > 0)[0]
    if len(nonzero) == 0:
        return []
    combined = tfidf_scores * rf_importances
    if is_phishing:
        top = nonzero[np.argsort(combined[nonzero])[::-1]][:5]
    else:
        top = nonzero[np.argsort(combined[nonzero])[::-1]][:5]
    return [feature_names[i] for i in top]

# ── Signal dictionaries ────────────────────────────────────
URGENCY = {'urgent', 'immediately', 'now', 'asap', 'quickly', 'expire',
           'expires', 'expiring', 'deadline', 'limited', 'hours', 'within',
           'today', 'tomorrow', 'minutes', 'before', 'friday', 'monday'}
THREAT  = {'suspended', 'blocked', 'locked', 'terminated', 'closed',
           'cancelled', 'deactivated', 'compromised', 'restricted',
           'freeze', 'frozen', 'cancelled', 'malpractice'}
ACTION  = {'click', 'verify', 'confirm', 'update', 'submit', 'login',
           'sign', 'enter', 'provide', 'send', 'fill', 'appeal', 'claim'}
SENSITIVE = {'password', 'pin', 'bvn', 'nin', 'account', 'card',
             'credit', 'bank', 'details', 'credentials', 'otp', 'token',
             'atm', 'number', 'ssn'}
REWARD  = {'won', 'winner', 'prize', 'reward', 'gift', 'free',
           'congratulations', 'selected', 'lucky', 'claim', 'bonus',
           'relief', 'fund', 'payment', 'ngn', '500', '000'}
URL_PATTERN = re.compile(r'http[s]?://\S+|www\.\S+')
LEGITIMATE = {'meeting', 'schedule', 'agenda', 'regards', 'attached',
              'document', 'seminar', 'assignment', 'lecture', 'department',
              'office', 'colleague', 'team', 'portal', 'registration',
              'session', 'hi', 'dear', 'hello', 'sincerely', 'best'}

def generate_explanation(email_text, result, confidence, keywords):
    words = set(email_text.lower().split())
    has_url = bool(URL_PATTERN.search(email_text))

    found_urgency  = words & URGENCY
    found_threat   = words & THREAT
    found_action   = words & ACTION
    found_sensitive = words & SENSITIVE
    found_reward   = words & REWARD
    found_legit    = words & LEGITIMATE

    if result == 'PHISHING':
        signals = []

        if has_url:
            url = URL_PATTERN.search(email_text).group()
            signals.append(f"an embedded suspicious URL ({url[:40]})")
        if found_threat:
            signals.append(f"account threat language ({', '.join(list(found_threat)[:2])})")
        if found_urgency:
            signals.append(f"urgency-creating language ({', '.join(list(found_urgency)[:2])})")
        if found_reward:
            signals.append(f"reward or financial enticement language ({', '.join(list(found_reward)[:2])})")
        if found_action and found_sensitive:
            signals.append(f"a request to {list(found_action)[0]} sensitive data ({', '.join(list(found_sensitive)[:2])})")
        elif found_sensitive:
            signals.append(f"references to sensitive personal data ({', '.join(list(found_sensitive)[:2])})")
        elif found_action:
            signals.append(f"suspicious calls to action ({', '.join(list(found_action)[:2])})")

        if not signals:
            kw = ', '.join(f'"{w}"' for w in keywords[:3])
            signals.append(f"linguistic patterns the model flagged, including {kw}")

        top2 = signals[:2]
        signal_text = ' combined with '.join(top2) if len(top2) == 2 else top2[0]

        if confidence >= 90:
            return (f"This email is highly likely to be a phishing attempt ({confidence}% confidence). "
                    f"It contains {signal_text} — a combination strongly associated with social engineering "
                    f"attacks designed to deceive recipients into disclosing sensitive information or taking harmful action.")
        elif confidence >= 75:
            return (f"This email shows significant phishing indicators ({confidence}% confidence). "
                    f"The presence of {signal_text} are patterns commonly used by attackers to manipulate "
                    f"recipients through false urgency, fear, or reward — do not click any links or provide personal data.")
        else:
            return (f"This email was flagged as potentially suspicious ({confidence}% confidence). "
                    f"It contains {signal_text}, which overlap with known phishing patterns. "
                    f"Verify the sender through official channels before responding.")

    else:
        if found_legit:
            legit_words = ', '.join(list(found_legit)[:3])
            return (f"This email appears legitimate ({confidence}% confidence). "
                    f"It contains professional or institutional language ({legit_words}) "
                    f"and does not exhibit the urgency, threats, suspicious URLs, or requests "
                    f"for sensitive data that characterise phishing emails.")
        elif confidence >= 80:
            return (f"This email appears legitimate ({confidence}% confidence). "
                    f"Its tone and content are consistent with normal communication and do not "
                    f"match the deceptive patterns — such as account threats, reward claims, or "
                    f"urgent requests for personal data — associated with phishing attacks.")
        else:
            return (f"This email is likely safe but borderline ({confidence}% confidence). "
                    f"It does not strongly match phishing patterns, however some ambiguous "
                    f"language is present — verify the sender before sharing sensitive information.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        word_count = len(text.split())
        SAFE_GREETINGS = {
            'hi', 'hey', 'hello', 'ok', 'okay', 'thanks', 'thank you',
            'yes', 'no', 'sure', 'noted', 'alright', 'bye', 'goodbye',
            'good morning', 'good afternoon', 'good evening'
        }
        if word_count < 8 or text.lower().strip() in SAFE_GREETINGS:
            return jsonify({
                'result': 'SAFE',
                'confidence': 50.0,
                'keywords': [],
                'explanation': 'This message is too short to analyse reliably. Please paste the full email content for an accurate phishing detection result.',
                'probability': 50.0
            })

        vec = tfidf_vectorizer.transform([text])
        prob_phishing = ensemble_model.predict_proba(vec)[0][1]
        is_phishing = prob_phishing >= THRESHOLD
        confidence = round(float(prob_phishing if is_phishing else 1 - prob_phishing) * 100, 2)
        result = 'PHISHING' if is_phishing else 'SAFE'
        keywords = get_keywords(vec, is_phishing)
        explanation = generate_explanation(text, result, confidence, keywords)

        save_prediction(text, result, confidence, keywords, explanation)

        return jsonify({
            'result': result,
            'confidence': confidence,
            'keywords': keywords,
            'explanation': explanation,
            'probability': round(float(prob_phishing) * 100, 2)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

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
            'id': r[0], 'text': r[1], 'result': r[2],
            'confidence': r[3],
            'keywords': [k.strip() for k in r[4].split(',')] if r[4] else [],
            'explanation': r[5], 'timestamp': r[6]
        } for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'version': '8.0'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

init_db()