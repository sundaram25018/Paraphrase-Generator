from flask import Flask, request, render_template
from markupsafe import Markup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
import re
from nltk.tokenize import sent_tokenize
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__)
nltk.download("punkt")

# Constants
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE == "cuda":
    model.half()
model.eval()  # Set model to evaluation mode

# Helper function to make URLs clickable
def linkify(text):
    url_pattern = r'(https?://[^\s]+)'
    linked = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    return Markup(linked)  # Treat as safe HTML

@torch.no_grad()
def paraphrase_chunk(text, max_input_length=256, max_output_length=256, top_k=30, top_p=0.85):
    text = text.strip()
    tokens = tokenizer.encode(text, truncation=True, max_length=max_input_length)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"paraphrase: {truncated_text} </s>"
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_input_length).to(DEVICE)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=0.8,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

@lru_cache(maxsize=32)
@torch.no_grad()
def paraphrase_full_text(paragraph):
    sentences = sent_tokenize(paragraph)
    prompts = [f"paraphrase: {sent.strip()} </s>" for sent in sentences]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        num_return_sequences=1,
        do_sample=True,
        top_k=30,
        top_p=0.85,
        temperature=0.8,
    )
    paraphrased_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return " ".join(paraphrased_sentences)

@torch.no_grad()
def paraphrase_partial_fixed(paragraph):
    words = paragraph.strip().split()
    if len(words) <= 15:
        return paraphrase_full_text(paragraph)

    start_index = 0
    end_index = 15
    to_paraphrase = " ".join(words[start_index:end_index])
    after = " ".join(words[end_index:])

    prompt = f"paraphrase: {to_paraphrase} </s>"
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(DEVICE)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        num_return_sequences=1,
        do_sample=True,
        top_k=30,
        top_p=0.85,
        temperature=0.8,
    )

    paraphrased_part = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return f"{paraphrased_part} {after}".strip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    full_text = ""
    partial_text = ""
    action = ""

    if request.method == "POST":
        full_text = request.form.get("full_text", "").strip()
        partial_text = request.form.get("partial_text", "").strip()
        action = request.form.get("action")

        if action == "full" and full_text:
            result = paraphrase_full_text(full_text)
        elif action == "partial" and partial_text:
            result = paraphrase_partial_fixed(partial_text)

        result = linkify(result)

    return render_template("index.html", result=result, full_text=full_text, partial_text=partial_text, action=action)

if __name__ == "__main__":
    app.run(debug=True)