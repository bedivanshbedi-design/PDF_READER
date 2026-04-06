import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# load model, cache for speed

@st.cache_resource
def load_models():
  embed_model = SentenceTransformer("all-MiniLM-L6-v2")
  qa_pipeline = pipeline(
      "text-generation",
      model = "google/flan-t5-small",
      max_new_tokens=80,
  #   temperature=0.1,
      do_sample=True
  )
  return embed_model, qa_pipeline

embed_model, qa_pipeline = load_models()

# Clean docuemnt  (Added after got wrong messy answer)

import re

## This is old clean text function not worked properly
# def clean_text(text):
#   # Fix broken words like "dat a" -> "data"
#   text = re.sub(r'(\w)\s+(\w)',r'\1\2', text)

#   # Remove multiple spaces
#   text =re.sub(r'\s+', ' ',text)

#   return text

# New clean text function

def clean_text(text):
  # Remove weird line breaks
  text = text.replace("\n", " ")

  # fix spacing issues
  text = re.sub(r'\s+', ' ',text)

  # Remove URLS
  text = re.sub(r'http\S+|www\S+\.com', '',text)

  return text.strip()

# PDF Preprocessing
def load_pdf(file):
  doc = fitz.open(stream=file.read(), filetype="pdf")
  text=" "
  for page in doc:
    text += page.get_text()
  return clean_text(text)  # Clean here


def chunk_text(text, chunk_size=150, overlap=50):
  words = text.split()
  chunks = []
  for i in range(0,len(words), chunk_size-overlap):
     chunk = " ".join(words[i:i+chunk_size])
     chunks.append(chunk)

  return chunks


# Vector Store

def create_index(chunks):
  embeddings = embed_model.encode(chunks)
  embeddings = np.array(embeddings)

  index = faiss.IndexFlatL2(embeddings.shape[1])
  index.add(embeddings)

  return index

# Adding new relevant chunk function here(only good chunks should be retrieved)
def get_relevant_chunks(query,index, chunks, k=5):
  query_embedding = embed_model.encode([query])
  distances, indices = index.search(np.array(query_embedding),k)

  selected_chunks = []
  for i in indices[0]:
    chunk = chunks[i]
    if 30 < len(chunk.split()) < 120:
      selected_chunks.append(chunk)

  return selected_chunks[:2]


def ask_question(query,index,chunks):
  # New retrieval
  relevant_chunk = get_relevant_chunks(query,index,chunks)

  context =" ".join(relevant_chunk)
  context = clean_text(context)

  # updated prompt because wise precise and 10 lines answer only news lines added
  prompt = f"""
  -Answer only from the context
  - Do NOT repeat context
  - Do NOT explain extra
  - Maximum 3 lines
  - If answer not found say: not in document

  Context:
  {context}

  Question:
  {query}

  Answer:
  """

  answer = qa_pipeline(prompt)[0]["generated_text"]
  return answer, context

# UI

st.set_page_config(page_title="PDF Chat",layout="centered")

st.title("Chat with yout PDF")

uploaded_file = st.file_uploader("Upload PDF", type ="pdf")

if uploaded_file:
  text = load_pdf(uploaded_file)
  chunks=chunk_text(text)
  index = create_index(chunks)

  st.success("PDF Processed")

  if "chat" not in st.session_state:
    st.session_state.chat = []

  query =st.chat_input("Ask something...")

  if query:
    answer, context = ask_question(query,index,chunks)

    st.session_state.chat.append(("user",query))
    st.session_state.chat.append(("bot",answer))
    st.session_state.context = context

  for role,msg in st.session_state.chat:
    with st.chat_message(role):
      st.write(msg)

  if "context" in st.session_state:
    with st.expander ("context used"):
      st.write(st.session_state.context)








