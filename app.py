from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import os
import gdown
import shap
import traceback

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://phishguard-gamma-ten.vercel.app",
])

THRESHOLD = 0.70

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
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
print("Model loaded!")

# ── Build SHAP explainers ONCE at startup (not per-request -- slow) ──
rf_model = ensemble_model.named_estimators_['rf']
xgb_model = ensemble_model.named_estimators_['xgb']
lr_model = ensemble_model.named_estimators_['lr']
ENSEMBLE_WEIGHTS = dict(zip([name for name, _ in ensemble_model.estimators], ensemble_model.weights))
TOTAL_WEIGHT = sum(ENSEMBLE_WEIGHTS.values())

print("Building SHAP explainers...")
rf_explainer = shap.TreeExplainer(rf_model)
xgb_explainer = shap.TreeExplainer(xgb_model)
print("SHAP explainers ready!")

def get_shap_contributions(vec):
    """Returns [(word, weighted_contribution), ...] for words present in
    the email, combining RF+LR+XGB using the SAME weights the
    VotingClassifier uses. Positive = pushes toward phishing."""
    vec_dense = vec.toarray()

    rf_shap = rf_explainer.shap_values(vec_dense)
    rf_contrib = rf_shap[1][0] if isinstance(rf_shap, list) else (
        rf_shap[0][:, 1] if rf_shap.ndim == 3 else rf_shap[0]
    )

    xgb_shap = xgb_explainer.shap_values(vec_dense)
    xgb_contrib = xgb_shap[1][0] if isinstance(xgb_shap, list) else (
        xgb_shap[0][:, 1] if xgb_shap.ndim == 3 else xgb_shap[0]
    )

    lr_contrib = vec_dense[0] * lr_model.coef_[0]

    combined = (
        ENSEMBLE_WEIGHTS['rf'] * rf_contrib +
        ENSEMBLE_WEIGHTS['lr'] * lr_contrib +
        ENSEMBLE_WEIGHTS['xgb'] * xgb_contrib
    ) / TOTAL_WEIGHT

    nonzero_idx = vec.nonzero()[1]
    contributions = [(feature_names[i], float(combined[i])) for i in nonzero_idx]
    contributions.sort(key=lambda x: -x[1])
    return contributions

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

# ── Signal categories for the narrative explanation ────────
URGENCY = {'urgent', 'immediately', 'now', 'today', 'asap', 'quickly', 'fast', 'expire', 'expires', 'expiring', 'deadline', 'limited', 'hours', 'minutes'}
THREAT = {'suspended', 'blocked', 'locked', 'terminated', 'closed', 'cancelled', 'deactivated', 'compromised', 'restricted', 'freeze', 'frozen'}
ACTION = {'click', 'verify', 'confirm', 'update', 'submit', 'login', 'sign', 'enter', 'provide', 'send', 'fill'}
SENSITIVE = {'password', 'pin', 'bvn', 'nin', 'account', 'card', 'credit', 'bank', 'details', 'credentials', 'otp', 'token'}
REWARD = {'won', 'winner', 'prize', 'reward', 'gift', 'free', 'congratulations', 'selected', 'lucky', 'claim', 'bonus'}
LEGITIMATE = {'meeting', 'schedule', 'agenda', 'regards', 'attached', 'document', 'seminar', 'assignment', 'lecture', 'department', 'office', 'colleague', 'team', 'portal', 'registration', 'session'}

def generate_explanation(email_text, result, confidence, contributions):
    """Generate a specific explanation grounded in the words that ACTUALLY
    drove the ensemble's decision (SHAP contributions), not just any
    category word appearing anywhere in the text."""
    words = set(email_text.lower().split())

    found_urgency = words & URGENCY
    found_threat = words & THREAT
    found_action = words & ACTION
    found_sensitive = words & SENSITIVE
    found_reward = words & REWARD
    found_legit = words & LEGITIMATE

    # Top words that actually pushed the decision, in the direction of the result
    top_phishing_words = [w for w, s in contributions if s > 0][:3]
    top_safe_words = [w for w, s in contributions if s < 0][:3]

    if result == 'PHISHING':
        signals = []
        if found_urgency:
            signals.append(f"urgency-creating language ({', '.join(list(found_urgency)[:2])})")
        if found_threat:
            signals.append(f"account threat language ({', '.join(list(found_threat)[:2])})")
        if found_reward:
            signals.append(f"reward or prize language ({', '.join(list(found_reward)[:2])})")
        if found_action and found_sensitive:
            signals.append(f"requests for sensitive action ({', '.join(list(found_action)[:1])} + {', '.join(list(found_sensitive)[:1])})")
        elif found_action:
            signals.append(f"suspicious calls to action ({', '.join(list(found_action)[:2])})")
        if found_sensitive:
            signals.append(f"references to sensitive information ({', '.join(list(found_sensitive)[:2])})")

        if not signals and top_phishing_words:
            signals.append(f'specific terms the model weighed heavily, including "{", ".join(top_phishing_words)}"')
        elif not signals:
            signals.append("overall linguistic patterns matching known phishing templates")

        signal_text = ' and '.join(signals[:2]) if len(signals) >= 2 else signals[0]

        if confidence >= 90:
            return f"This email is highly likely to be a phishing attempt. It contains {signal_text}, which are strong indicators of social engineering designed to deceive the recipient into taking harmful action. Do not interact with any links or attachments."
        elif confidence >= 75:
            return f"This email shows significant phishing characteristics. The presence of {signal_text} are patterns commonly used by attackers to create false urgency or fear, manipulating recipients into disclosing sensitive information or clicking malicious links."
        else:
            return f"This email was flagged as potentially suspicious due to {signal_text}. While the confidence is moderate at {confidence}%, these patterns are associated with phishing attempts — verify the sender's identity through official channels before responding."

    else:
        if found_legit:
            legit_text = ', '.join(list(found_legit)[:3])
            return f"This email appears legitimate. It contains contextually appropriate institutional or professional language ({legit_text}) without the urgency, threats, or suspicious calls to action typically found in phishing emails. The overall tone and content are consistent with genuine communication."
        elif top_safe_words:
            return f"This email appears legitimate with {confidence}% confidence. Terms like \"{', '.join(top_safe_words)}\" weighed most heavily toward a safe classification, and the message lacks the urgency, threats, or requests for sensitive data typically seen in phishing attempts."
        elif confidence >= 80:
            return f"This email appears legitimate with {confidence}% confidence. Its language and structure are consistent with normal communication patterns and do not exhibit the deceptive characteristics associated with phishing attempts."
        else:
            return f"This email is likely safe but the classification is borderline at {confidence}% confidence. It does not strongly match phishing patterns, however some ambiguous language was detected. Verify the sender's identity before sharing any sensitive information or clicking links."

# ── Routes ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        word_count = len(text.split())
        SAFE_GREETINGS = {'hi', 'hey', 'hello', 'ok', 'okay', 'thanks', 'thank you', 'yes', 'no', 'sure', 'noted', 'alright', 'bye', 'goodbye', 'good morning', 'good afternoon', 'good evening'}
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

        contributions = get_shap_contributions(vec)
        keywords = [w for w, s in (contributions if is_phishing else contributions[::-1])][:5]
        explanation = generate_explanation(text, result, confidence, contributions)

        save_prediction(text, result, confidence, keywords, explanation)

        return jsonify({
            'result': result,
            'confidence': confidence,
            'keywords': keywords,
            'explanation': explanation,
            'probability': round(float(prob_phishing) * 100, 2)
        })
    except Exception as e:
        # TEMPORARY: return the real error so we can see exactly what broke.
        # Remove the 'trace' field once this is confirmed working.
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

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
    return jsonify({'status': 'running', 'version': '7.0'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

init_db()