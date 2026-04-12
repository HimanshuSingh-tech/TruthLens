import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load both files
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# 2. Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# 3. Combine them and keep only what we need
df = pd.concat([true_df, fake_df])[['title', 'label']]

# 4. Text to Numbers (TF-IDF)
# This is the "Applied Statistical Skill" recruiters want to see!
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['title'])
y = df['label']

# 5. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Save the Brain and the Dictionary
joblib.dump(model, 'news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("✅ Success! 'news_model.pkl' created. Your AI is now trained.")