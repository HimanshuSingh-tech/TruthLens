import requests
from app import get_credibility

url = 'https://newsapi.org/v2/top-headlines?country=us&pageSize=10&apiKey=164c9c624f4c46cc827d93fe6f135e17'
data = requests.get(url).json().get('articles', [])
with open("debug.txt", "w", encoding="utf-8") as f:
    for a in data:
        src = a.get('source', {}).get('name', 'Unknown')
        title = a.get('title', '')
        cred = get_credibility(title, src, 0.0)
        f.write(f"[{cred['score']:3d}] {src:20} -> {cred['label']} | {cred['reasons']}\n")
