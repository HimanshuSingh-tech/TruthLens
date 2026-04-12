# 🔍 TruthLens: AI-Powered News Credibility Detector

**TruthLens** is a full-stack Machine Learning application designed to combat misinformation. It aggregates real-time news and uses Natural Language Processing (NLP) to provide a credibility score for any headline.

---

### 🌟 Key Features
* **Live News Feed:** Fetches the latest global news using NewsAPI.
* **AI Analysis:** Uses a trained **Logistic Regression** model to detect fake news patterns.
* **Interactive Checker:** A custom "Analyse" tool to check any headline for truthfulness.
* **Modern UI:** A sleek, dark-themed dashboard built with Flask and Glassmorphism CSS.

### 🛠️ Tech Stack
* **Backend:** Python (Flask)
* **Machine Learning:** Scikit-Learn, TF-IDF Vectorization
* **NLP:** TextBlob (Sentiment Analysis), NLTK
* **Frontend:** HTML5, CSS3, JavaScript

### 🧠 How It Works
The core model was trained on a dataset of over 40,000 news articles. By converting text into numerical data using **TF-IDF with Bi-grams**, the AI identifies sensationalist language and structural patterns typical of misinformation.

---

### 🚀 Setup & Installation
1. Clone the repository:
   `git clone https://github.com/HimanshuSingh-tech/TruthLens.git`
2. Install dependencies:
   `pip install -r requirement.txt`
3. Run the app:
   `python app.py`
