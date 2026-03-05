#qa_bot.py
import os
import torch
import numpy as np
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from utils.pdf_reader import extract_text_from_pdf
from utils.text_splitter import split_text
from embedder import embed_text_chunks

# ✅ Load environment and models
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load & process PDF
PDF_PATH = "data/sample.pdf"
text = extract_text_from_pdf(PDF_PATH)
chunks = split_text(text)
embeddings = embed_text_chunks(chunks)
embedding_tensor = torch.tensor(embeddings)

print("✅ PDF loaded and embedded. Ask your questions! (type 'exit' to quit)\n")

while True:
    question = input("Your Question: ").strip()
    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Get question embedding and cosine scores
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embedding_tensor)[0]

    # Get top 3 chunks
    top_k = 3
    top_indices = torch.topk(scores, top_k).indices.tolist()
    top_chunks = [chunks[i] for i in top_indices]
    context = "\n\n---\n\n".join(top_chunks)

    # Generate response
    prompt = f"""You're a helpful assistant. Use only the following context to answer the user's question.

Context:
{context}

Question:
{question}

Answer in a helpful and concise way."""

    response = gemini_model.generate_content(prompt)
    print("\nAnswer:", response.text.strip(), "\n")
