import os
import re
import unicodedata
import numpy as np
import faiss
import torch
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pypdf import PdfReader
import json

app = Flask(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PDF_DIR    = os.environ.get("PDF_DIR", "./books")
INDEX_PATH = os.environ.get("INDEX_PATH", "./rag_index")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_ID      = os.environ.get("LLM_ID", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
embedder  = None
reranker  = None
tokenizer = None
model     = None
index     = None
all_chunks = []
metas      = []

# ─── TEXT UTILS ──────────────────────────────────────────────────────────────
CTRL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")

def clean_for_tokenizer(s: str) -> str:
    s = s.replace("\x00", " ")
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Cs": continue
        if cat == "Cc" and ch not in ("\n", "\t"): continue
        if cat.startswith("C") and cat not in ("Cc",): continue
        out.append(ch)
    s = "".join(out)
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_text(text: str, chunk_size=1400, overlap=250):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if len(chunk) > 200:
            chunks.append(chunk)
        start = end - overlap
        if start < 0: start = 0
        if end == n: break
    return chunks

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").replace("\x00", " ").strip()
        if txt:
            parts.append(f"\n\n[PAGE {i+1}]\n{txt}")
    return "\n".join(parts)

# ─── INDEX BUILD / LOAD ───────────────────────────────────────────────────────
def build_index():
    global all_chunks, metas, index
    os.makedirs(INDEX_PATH, exist_ok=True)
    chunks_path = os.path.join(INDEX_PATH, "chunks.json")
    index_path  = os.path.join(INDEX_PATH, "faiss.index")

    if os.path.exists(chunks_path) and os.path.exists(index_path):
        print("Loading cached index…")
        with open(chunks_path) as f:
            data = json.load(f)
        all_chunks = data["chunks"]
        metas      = data["metas"]
        index      = faiss.read_index(index_path)
        print(f"Loaded {index.ntotal} vectors.")
        return

    print("Building index from PDFs…")
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    docs = []
    for p in pdf_files:
        text = extract_text_from_pdf(p)
        docs.append({"source": os.path.basename(p), "text": text})

    all_chunks, metas = [], []
    for d in docs:
        for j, c in enumerate(chunk_text(d["text"])):
            all_chunks.append(clean_for_tokenizer(c))
            metas.append({"source": d["source"], "chunk_id": j})

    embs = embedder.encode(all_chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    with open(chunks_path, "w") as f:
        json.dump({"chunks": all_chunks, "metas": metas}, f)
    faiss.write_index(index, index_path)
    print(f"Index built: {index.ntotal} vectors.")

# ─── RAG PIPELINE ────────────────────────────────────────────────────────────
def retrieve(query: str, top_k=30):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({"score": float(score), "text": all_chunks[idx], "meta": metas[idx]})
    return results

def rerank_hits(query, hits, top_k=8):
    pairs = [(query, h["text"]) for h in hits]
    scores = reranker.predict(pairs)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return hits[:top_k]

def build_context(hits, max_chars=9000):
    ctx_parts, total = [], 0
    for i, h in enumerate(hits, 1):
        tag = f"[{i}] Source={h['meta']['source']} Chunk={h['meta']['chunk_id']} Score={h['score']:.3f}"
        block = f"{tag}\n{h['text'].strip()}\n"
        if total + len(block) > max_chars:
            break
        ctx_parts.append(block)
        total += len(block)
    return "\n\n".join(ctx_parts)

def generate_answer(query: str, top_k=8, retrieve_k=30, max_new_tokens=400):
    hits = retrieve(query, top_k=retrieve_k)
    hits = rerank_hits(query, hits, top_k=top_k)
    context = build_context(hits)

    messages = [
        {"role": "system", "content": (
            "You are a helpful AI assistant. Use ONLY the provided context from books. "
            "If the answer is not in the context, say you don't know. "
            "When you use a fact, cite it like [1], [2] matching the context blocks."
        )},
        {"role": "user", "content": (
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            "Now write a single combined answer that merges relevant info from multiple sources, "
            "and includes citations."
        )},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.05,
        )

    prompt_len = inputs["input_ids"].size(-1)
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text, hits

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index_page():
    books = []
    if os.path.exists(PDF_DIR):
        books = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    return render_template("index.html", books=books, model_loaded=(model is not None))

@app.route("/ask", methods=["POST"])
def ask():
    if model is None:
        return jsonify({"error": "Model not loaded yet. Please wait."}), 503
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer_text, hits = generate_answer(query)
        sources = [{"source": h["meta"]["source"], "chunk_id": h["meta"]["chunk_id"],
                    "score": round(h["rerank_score"], 3), "preview": h["text"][:200]}
                   for h in hits[:5]]
        return jsonify({"answer": answer_text, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    return jsonify({
        "model_loaded": model is not None,
        "index_built": index is not None,
        "chunks": len(all_chunks),
        "vectors": index.ntotal if index else 0,
        "device": DEVICE,
    })

# ─── STARTUP ─────────────────────────────────────────────────────────────────
def load_models():
    global embedder, reranker, tokenizer, model
    print("Loading embedder…")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    print("Loading reranker…")
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)

    build_index()

    print("Loading LLM…")
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)

    if DEVICE == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs = {"quantization_config": bnb_config, "torch_dtype": torch.float16}
    else:
        model_kwargs = {"torch_dtype": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, device_map="auto", **model_kwargs)
    print("All models ready ✓")

if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
