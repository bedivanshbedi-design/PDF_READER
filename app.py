import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

#Setup Logger

import logging

logging.basicConfig(
    level=logging.INFO,
    format ="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



# load model, cache for speed

@st.cache_resource
def load_models():
  embed_model = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
  qa_pipeline = pipeline(
      "text-generation",
      model = "google/flan-t5-small",
      max_new_tokens=80,
  #   temperature=0.1,
      do_sample=False                                                               
  )
  logger.info("Models loaded successfully")
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
  logger.info("starting PDF processing")
  doc = fitz.open(stream=file.read(), filetype="pdf")
  text=" "
  for page in doc:
    text += page.get_text()
  logger.info(f"Extracted text length:{len(text)}")
  return clean_text(text)  # Clean here


def chunk_text(text, chunk_size=150, overlap=50):
  words = text.split()
  chunks = []
  for i in range(0,len(words), chunk_size-overlap):
     chunk = " ".join(words[i:i+chunk_size])
     chunks.append(chunk)

  logger.info(f"Total chunks created:{len(chunks)}")
  if chunks:
    logger.info(f"sample chunk: {chunks[0][:200]}")
  return chunks


# Vector Store

def create_index(chunks):
  logger.info("Creating embeddings..")
  embeddings = embed_model.encode(chunks)
  embeddings = np.array(embeddings)

  logger.info(f"Embeddings shape: {embeddings.shape}")
  
  index = faiss.IndexFlatL2(embeddings.shape[1])
  index.add(embeddings)



  logger.info("FAISS index created successfully")
  return index


# Adding new relevant chunk function here(only good chunks should be retrieved)
def get_relevant_chunks(query,index, chunks, k=5):
  query_embedding = embed_model.encode([query])
  # distances, indices = index.search(np.array(query_embedding),k)
  
  scores = np.dot(embeddings, query_embedding)
  top_k_idx = np.argsort(scores)[-k:][::-1]

  return [chunks[i] for i in top_k_idx[:2]]

  # selected_chunks = []
  # for i in indices[0]:
  #   chunk = chunks[i]
  #   selected_chunks.append(chunk)

  logger.info(f"Selected chunks count: {len(selected_chunks)}")
  # return selected_chunks[:2]


def ask_question(query,index,chunks):
  # New retrieval
  relevant_chunk = get_relevant_chunks(query,index,chunks)

  context =" ".join(relevant_chunk)
  context = clean_text(context)

  logger.info(f"context length: {len(context)}")
  

  # updated prompt because wise precise and 10 lines answer only news lines added
  prompt = f"""
  Answer only from the context below.
  If answer not found say: not in document

  Context:
  {context}

  Question:
  {query}

  Answer:
  """

  result=qa_pipeline(prompt)
  answer = result[0]["generated_text"].replace(Prompt,"").strip()
  logger.info(f"final answer: {answer}")
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








