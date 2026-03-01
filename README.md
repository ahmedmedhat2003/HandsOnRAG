# Book RAG — Deploy on Railway

A beautiful web UI for your ML book RAG system, ready to deploy on Railway.

## Project Structure

```
rag-app/
├── app.py              # Flask backend + RAG pipeline
├── templates/
│   └── index.html      # UI (chat interface)
├── requirements.txt
├── Procfile
├── railway.json
├── .env.example
└── books/              # ← put your PDF files here
```

## Quick Deploy to Railway

### 1. Add your PDF books
Put all your PDF files inside the `books/` folder before deploying.

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "RAG app"
gh repo create my-rag-app --public --push
```

### 3. Deploy on Railway
1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub Repo
2. Select your repository
3. Add **Environment Variables** in Railway dashboard:

| Variable | Value |
|----------|-------|
| `HF_TOKEN` | Your HuggingFace token (needed for Llama) |
| `PDF_DIR` | `./books` |
| `INDEX_PATH` | `./rag_index` |

4. Railway will build and deploy automatically.

### 4. Access your app
Railway gives you a public URL like `https://my-rag-app.up.railway.app`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token (required for Llama) |
| `PDF_DIR` | `./books` | Directory with PDF files |
| `INDEX_PATH` | `./rag_index` | Where to cache the FAISS index |
| `LLM_ID` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID |
| `PORT` | `8080` | Auto-set by Railway |

---

## Notes

- **First startup** builds the FAISS index from your PDFs — this takes a few minutes
- The index is **cached** to `./rag_index` so subsequent starts are fast
- Railway's free tier may be too small for GPU inference — use a **GPU instance** or switch `LLM_ID` to a smaller model
- For CPU-only deployment, consider using the OpenAI API instead of a local LLM

## Changing the LLM

To use OpenAI instead of local Llama, replace the `generate_answer()` function in `app.py` with:

```python
import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_answer(query, top_k=8, retrieve_k=30, **kwargs):
    hits = retrieve(query, top_k=retrieve_k)
    hits = rerank_hits(query, hits, top_k=top_k)
    context = build_context(hits)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided context. Cite sources as [1], [2]."},
            {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer with citations:"}
        ]
    )
    return response.choices[0].message.content, hits
```
