import os
import gradio as gr
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Extract text safely
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file.name)  # <-- FIXED HERE
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def ask_pdf(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF."

    if question.strip() == "":
        return "Please enter a question."

    document_text = extract_pdf_text(pdf_file)

    # Limit text to avoid token overflow
    document_text = document_text[:6000]

    prompt = f"""
You are an assistant.
Answer the question using ONLY the document context below.

Context:
{document_text}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown("# 📄 LLM-Based PDF Query Engine")

    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    question_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        ask_pdf,
        inputs=[pdf_input, question_input],
        outputs=output
    )

demo.launch()
