"""
Smart News Aggregator + Fake News Detector
Flask backend with credibility scoring engine
"""

from flask import Flask, render_template, request, jsonify
import requests
from textblob import TextBlob
import nltk
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

NEWS_API_KEY = "164c9c624f4c46cc827d93fe6f135e17"

# ─────────────────────────────────────────────────────────────────────────────
# TRUSTED SOURCE WHITELIST  — exact + keyword lists for partial matching
# ─────────────────────────────────────────────────────────────────────────────
TRUSTED_SOURCES_EXACT = {
    "bbc news", "bbc sport", "reuters", "associated press", "ap news",
    "the guardian", "npr", "the new york times", "the washington post",
    "the hindu", "ndtv", "india today", "times of india", "hindustan times",
    "al jazeera english", "bloomberg", "the economist", "financial times",
    "cnn", "abc news", "cbs news", "nbc news", "pbs newshour", "axios",
    "politico", "the atlantic", "wired", "techcrunch", "the verge",
    "ars technica", "scientific american", "national geographic",
    "espn", "sky news", "cnbc", "afp", "business standard",
    "live mint", "the wire", "scroll.in", "the quint", "indian express",
    "deccan herald", "tribune", "the telegraph", "the independent",
    "time", "newsweek", "forbes", "fortune", "wall street journal",
    "the new yorker", "usa today", "los angeles times"
}

TRUSTED_KEYWORDS = [
    "nbc news", "cbs news", "pbs", "axios", "politico", "wired",
    "techcrunch", "the verge", "ars technica", "national geographic",
    "espn", "sky news", "cnbc", "ndtv", "india today", "indian express",
    "deccan herald", "live mint", "the wire", "scroll", "the quint",
    "newsweek", "forbes", "fortune", "wall street journal",
    "usa today", "los angeles times", "the hindu", "afp"
]

SUSPICIOUS_KEYWORDS = [
    "natural news", "infowars", "breitbart", "daily wire",
    "the blaze", "newsmax", "oann", "worldnews", "beforeitsnews",
    "zerohedge", "yournewswire", "newspunch", "activistpost"
]

# ─────────────────────────────────────────────────────────────────────────────
# CLICKBAIT & MANIPULATION PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
CLICKBAIT_WORDS = [
    r'\bshocking\b', r'\bexposed\b', r'you won\'t believe',
    r'\bsecret\b', r'\bmiracle\b', r'\bhoax\b', r'\bscam\b',
    r'\bconspiracy\b', r'\bcover.?up\b', r'\bdeep state\b',
    r'they don\'t want you', r'what they\'re hiding', r'the truth about',
    r'\bblown away\b', r'\bjaw.?drop\b', r'watch what happens',
    r'\bsuppressed\b', r'\bwake up\b', r'\bplandemic\b'
]

def count_caps_words(text):
    return sum(1 for w in text.split() if w.isupper() and len(w) > 2)

def count_clickbait(text):
    text_lower = text.lower()
    return sum(1 for p in CLICKBAIT_WORDS if re.search(p, text_lower))

def count_punctuation(text):
    return text.count('!') + text.count('?')

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE CLASSIFIER  (substring matching so "ESPN" hits "espn" keyword)
# ─────────────────────────────────────────────────────────────────────────────
def classify_source(src):
    s = (src or "").lower().strip()
    if not s:
        return "unknown"
    if any(k in s for k in SUSPICIOUS_KEYWORDS):
        return "suspicious"
    if s in TRUSTED_SOURCES_EXACT:
        return "trusted"
    if any(k in s for k in TRUSTED_KEYWORDS):
        return "trusted"
    return "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# CREDIBILITY SCORER
# Thresholds: >=65 Real | 40-64 Suspicious | <40 Likely Fake
# Baseline 50 (neutral).  Trusted +30, suspicious -35.
# Clean headline bonuses ensure legitimate news from known sources hits Real.
# Red-flag deductions push clickbait/caps articles into Suspicious/Fake.
# ─────────────────────────────────────────────────────────────────────────────
import joblib

# Load the AI models at the top of app.py
try:
    ai_model = joblib.load('news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except:
    ai_model = None


def get_credibility(title, source_name, sentiment_score):
    reasons = []

    # 1. Start with AI Prediction
    if ai_model and vectorizer:
        transformed_title = vectorizer.transform([title])
        prediction = ai_model.predict(transformed_title)[0]
        probability = ai_model.predict_proba(transformed_title).max() * 100

        score = probability
        badge = "real" if prediction == "REAL" else "fake"
        label = "Likely Real" if prediction == "REAL" else "Likely Fake"
        color = "#00C896" if prediction == "REAL" else "#EF4444"
        reasons.append(f"🤖 AI Confidence: {round(probability, 1)}%")
    else:
        # Fallback if model is missing
        score, label, color, badge = 50, "AI Offline", "#F59E0B", "suspicious"
        reasons.append("⚠️ AI Model not loaded")

    # 2. Add Source Logic (The Human Element)
    src_type = classify_source(source_name)
    if src_type == "trusted":
        score = min(score + 10, 100)  # Give a small boost for known good sources
        reasons.append("✅ Verified & trusted source")
    elif src_type == "suspicious":
        score = max(score - 30, 0)  # Heavy penalty for suspicious sources
        reasons.append("🚫 Known unreliable source")

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "badge": badge,
        "reasons": reasons
    }

# ─────────────────────────────────────────────────────────────────────────────
# AI SUMMARY (newspaper3k)
# ─────────────────────────────────────────────────────────────────────────────
def get_summary(article_url, fallback):
    try:
        from newspaper import Article, Config
        config = Config()
        config.browser_user_agent = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        config.request_timeout = 6
        art = Article(article_url, config=config)
        art.download()
        art.parse()
        art.nlp()
        if art.summary and len(art.summary) > 30:
            return art.summary[:180] + "..."
    except Exception:
        pass
    return fallback or "Click 'Read Full Article' for details."

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ARTICLES
# ─────────────────────────────────────────────────────────────────────────────
def process_articles(raw_articles, max_summary=6):
    processed = []
    for i, article in enumerate(raw_articles):
        title = article.get('title') or ''
        if not title or title == '[Removed]':
            continue

        # Sentiment
        blob = TextBlob(title)
        score = blob.sentiment.polarity
        if score > 0.1:
            sentiment = "Positive 😊"
            sentiment_badge = "positive"
        elif score < -0.1:
            sentiment = "Negative 😟"
            sentiment_badge = "negative"
        else:
            sentiment = "Neutral 😐"
            sentiment_badge = "neutral"

        # AI Summary (only first N articles to avoid slow loads)
        if i < max_summary:
            summary = get_summary(
                article.get('url', ''),
                article.get('description', '')
            )
        else:
            summary = article.get('description') or "Click 'Read Full Article' for details."

        # Credibility
        source_name = (article.get('source') or {}).get('name', '')
        cred = get_credibility(title, source_name, score)

        processed.append({
            "title": title,
            "url": article.get('url', '#'),
            "urlToImage": article.get('urlToImage', ''),
            "source": source_name,
            "publishedAt": (article.get('publishedAt') or '')[:10],
            "sentiment": sentiment,
            "sentiment_badge": sentiment_badge,
            "summary": summary,
            "credibility": cred,
        })

    return processed

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    query = request.args.get('q', '').strip()

    if query:
        url = (f"https://newsapi.org/v2/everything?q={query}"
               f"&sortBy=relevancy&pageSize=12&apiKey={NEWS_API_KEY}")
    else:
        url = (f"https://newsapi.org/v2/top-headlines?country=us"
               f"&pageSize=12&apiKey={NEWS_API_KEY}")

    try:
        resp = requests.get(url, timeout=8)
        raw = resp.json().get('articles', [])
    except Exception:
        raw = []

    articles = process_articles(raw)
    return render_template('index.html', articles=articles, query=query)


@app.route('/check', methods=['POST'])
def check_headline():
    """API endpoint for manual headline credibility check."""
    data = request.get_json()
    headline = (data or {}).get('headline', '').strip()
    if not headline:
        return jsonify({"error": "No headline provided"}), 400

    blob = TextBlob(headline)
    sentiment_score = blob.sentiment.polarity
    cred = get_credibility(headline, '', sentiment_score)
    cred['sentiment_score'] = round(sentiment_score, 3)
    return jsonify(cred)


if __name__ == "__main__":
    app.run(debug=True)