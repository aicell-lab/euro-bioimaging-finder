# Euro-BioImaging Search Index

This repository hosts the Euro-BioImaging search index for public access.

## 📥 Access the Index

- **JSON Index**: [eurobioimaging_index.json](eurobioimaging_index.json)
- **BM25 Index**: [eurobioimaging_bm25_index.pkl](eurobioimaging_bm25_index.pkl)
- **Web Interface**: [index.html](index.html)

## 🔗 Direct Links

- JSON API: `https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json`
- BM25 Index: `https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl`
- Web Interface: `https://oeway.github.io/euro-bioimaging-finder/`

## 📊 Example Usage

```python
import pickle
import bm25s
import httpx
import json

# Download and load main index with metadata
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json')
combined_data = response.json()
bm25_metadata = combined_data.get('bm25_metadata', [])

# Download and load BM25 retriever
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl')
with open('eurobioimaging_bm25_index.pkl', 'wb') as f:
    f.write(response.content)

with open('eurobioimaging_bm25_index.pkl', 'rb') as f:
    retriever = pickle.load(f)

def fulltext_search(query, k=5):
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=k)
    hits = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = scores[0, i]
        metadata = bm25_metadata[doc_idx]
        hit = metadata.copy()
        hit["score"] = float(score)
        hits.append(hit)
    return hits

# Example search
results = fulltext_search("super resolution microscopy", k=5)
for hit in results:
    print(f"Score: {hit['score']:.2f} - Type: {hit['type']} - ID: {hit['id']}")
```

## 📊 Current Statistics

Last updated: 2025-06-28

## 🔬 About Euro-BioImaging

Euro-BioImaging is the European research infrastructure for biological and biomedical imaging.

Visit: [https://www.eurobioimaging.eu/](https://www.eurobioimaging.eu/)
