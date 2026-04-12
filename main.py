import requests
def fetch_top_news():
    api_key = "164c9c624f4c46cc827d93fe6f135e17"
    url = f"https://newsapi.org/v2/top-headlines?country=us&apikey={api_key}"
    print("News fetch ho rahi h.... thoda wait karo...")
    try:
        response = requests.get(url)
        data = response.json()

        articles = data.get('articles',[])
        if not articles:
            print("---No articles found API galat h---")
            print(data)
            return
        print("\n--- AAJ KI TOP HEADLINES ---")
        for i, article in enumerate(articles[:5],1):
            print(f"{i}. {article['title']}")
            print(f"{i}. {article['source']['name']}")
            print("-" * 40)
    except Exception as e:
        print(f"Error aa gaya bro: {e}")
if __name__ == "__main__":
    fetch_top_news()