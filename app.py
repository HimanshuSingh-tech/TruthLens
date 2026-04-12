"""
TruthLens: AI-Powered News Credibility Engine
Developed by: Himanshu Singh Gahlot
"""

import os
import re
import joblib
import requests
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import nltk

# Initialize Flask
app = Flask(__name__)

# Security: Fetch API Key from Environment Variables (set this in Vercel!)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Load our AI models
try:
    ai_model = joblib.load('news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    print(f"⚠️ Model Load Warning: {e}")
    ai_model = None
    vectorizer = None

# -----------------------------------------------------------------------------
# SOURCE VERIFICATION LISTS
# -----------------------------------------------------------------------------
TRUSTED_SOURCES = {
    "bbc news", "reuters", "associated press", "ap news", "the guardian",
    "the new york times", "the washington post", "the hindu", "ndtv",
    "india today", "times of india", "hindustan times", "bloomberg",
    "the economist", "financial times", "cnn", "abc news", "cbs news",
    "nbc news", "wired", "techcrunch", "the verge", "scientific american",
    "espn", "sky news", "cnbc", "afp", "live mint", "the wire", "scroll.in"
}

SUSPICIOUS_SOURCES = [
    "natural news", "infowars", "breitbart", "beforeitsnews",
    "zerohedge", "newspunch", "activistpost"
]

def classify_source(src):
    s = (src or "").lower().strip()
    if not s: return "unknown"
    if any(k in s for k in SUSPICIOUS_SOURCES): return "suspicious"
    if s in TRUSTED_SOURCES or any(k in s for k in TRUSTED_SOURCES): return "trusted"
    return "unknown"

# -----------------------------------------------------------------------------
# THE BRAINS: CREDIBILITY SCORER
# -----------------------------------------------------------------------------
def get_credibility(title, source_name):
    reasons = []

    # Base AI Logic
    if ai_model and vectorizer:
        transformed_title = vectorizer.transform([title])
        prediction = ai_model.predict(transformed_title)[0]
        probability = ai_model.predict_proba(transformed_title).max() * 100

        # Logic Fix: If AI says it's FAKE, the score should be low.
        if prediction == "FAKE":
            score = 100 - probability
            label = "Likely Fake"
            color = "#EF4444" # Red
            badge = "fake"
        else:
            score = probability
            label = "Likely Real"
            color = "#00C896" # Green
            badge = "real"

        reasons.append(f"🤖 AI Confidence: {round(probability, 1)}%")
    else:
        score, label, color, badge = 50, "AI Offline", "#F59E0B", "suspicious"
        reasons.append("⚠️ AI Model not loaded")

    # Human Element: Source Reputation
    src_type = classify_source(source_name)
    if src_type == "trusted":
        score = min(score + 15, 100) # Boost for known good sources
        reasons.append("✅ Verified & Trusted Source")
    elif src_type == "suspicious":
        score = max(score - 35, 0) # Heavy penalty for clickbait farms
        reasons.append("🚫 Known Unreliable Source")

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "badge": badge,
        "reasons": reasons
    }

# -----------------------------------------------------------------------------
# DATA PROCESSING
# -----------------------------------------------------------------------------
def process_articles(raw_articles):
    processed = []
    for article in raw_articles:
        title = article.get('title') or ''
        if not title or title == '[Removed]': continue

        # Simple Sentiment Analysis
        blob = TextBlob(title)
        pol = blob.sentiment.polarity
        if pol > 0.1: sent, s_badge = "Positive 😊", "positive"
        elif pol < -0.1: sent, s_badge = "Negative 😟", "negative"
        else: sent, s_badge = "Neutral 😐", "neutral"

        # Run Credibility Check
        source_name = (article.get('source') or {}).get('name', '')
        cred = get_credibility(title, source_name)

        processed.append({
            "title": title,
            "url": article.get('url', '#'),
            "urlToImage": article.get('urlToImage', ''),
            "source": source_name,
            "publishedAt": (article.get('publishedAt') or '')[:10],
            "sentiment": sent,
            "sentiment_badge": s_badge,
            "summary": article.get('description') or "Click 'Read Full Article' for more details.",
            "credibility": cred,
        })
    return processed

# -----------------------------------------------------------------------------
# WEB ROUTES
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    query = request.args.get('q', '').strip()

    # If search is used, find relevant news. Otherwise, show top US headlines.
    if query:
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=relevancy&pageSize=12&apiKey={NEWS_API_KEY}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=12&apiKey={NEWS_API_KEY}"

    try:
        resp = requests.get(url, timeout=8)
        raw = resp.json().get('articles', [])
    except:
        raw = []

    articles = process_articles(raw)
    return render_template('index.html', articles=articles, query=query)

@app.route('/check', methods=['POST'])
def check_headline():
    """Manual check for custom headlines from the search bar."""
    data = request.get_json()
    headline = (data or {}).get('headline', '').strip()
    if not headline:
        return jsonify({"error": "No headline provided"}), 400

    cred = get_credibility(headline, "User Input")
    return jsonify(cred)

if __name__ == "__main__":
    app.run(debug=True)