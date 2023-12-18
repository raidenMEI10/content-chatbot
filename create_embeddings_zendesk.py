import pickle
import requests

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

if __name__ == '__main__':
    response = requests.get('https://support.strikingly.com/api/v2/help_center/en-us/articles.json')
    articles = response.json().get('articles', [])
    pages = [{"text": article['body'], "source": article['html_url']} for article in articles]

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
