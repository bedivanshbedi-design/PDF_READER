import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from evaluate import run_evaluate


# if "eval_data" not in st.session_state:
#   st.session_state.eval_data =[]
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
      model = "gpt2",
      max_new_tokens=100,
      temperature=0.7,
      do_sample=True,
      top_p=0.1
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

  # index = faiss.IndexFlatL2(embeddings.shape[1])
  # index.add(embeddings)



  logger.info("FAISS index created successfully")
  return embeddings

uploaded_file = st.file_uploader("Upload PDF", type ="pdf",accept_multiple_files=True)

if uploaded_file:
  all_text =""
  for file in uploaded_file:
    logger.info(f"Processing file: {file.name}")
    text = load_pdf(file)
    all_text +=text + " "

  chunks=chunk_text(all_text)
  embeddings = create_index(chunks)

# Adding new relevant chunk function here(only good chunks should be retrieved)
def get_relevant_chunks(query,embeddings, chunks, k=5):
  query_embedding = embed_model.encode(query)
  # distances, indices = index.search(np.array(query_embedding),k)

  scores = np.dot(embeddings, query_embedding.T)
  top_k_idx = np.argsort(scores)[-k:][::-1]

  return [chunks[i] for i in top_k_idx[:k]]

  # selected_chunks = []
  # for i in indices[0]:
  #   chunk = chunks[i]
  #   selected_chunks.append(chunk)

  logger.info(f"Selected chunks count: {len(selected_chunks)}")
  # return selected_chunks[:2]


def ask_question(query,embeddings,chunks):
  # New retrieval
  relevant_chunk = get_relevant_chunks(query,embeddings,chunks)

  context =" ".join(relevant_chunk)
  context = clean_text(context)

  logger.info(f"context length: {len(context)}")


  # updated prompt because wise precise and 10 lines answer only news lines added
  prompt = f"""
  answer the question only from context below
  if answer not found, say 'Not in document'.
  Context:
  {context}

  Question:
  {query}

  Answer:
  """

  answer =""
  result=qa_pipeline(prompt)
  answer = result[0]["generated_text"][len(prompt):].strip()
  logger.info(f"final answer: {answer}")
  return answer, context

# UI

st.set_page_config(page_title="PDF Chat",layout="centered")

st.title("Chat with yout PDF")

if st.button("clean chat"):
  st.session_state.chat = []
  st.session_state.eval_data = []
  st.session_state.pop("context", None)




# if "context" in st.session_state:
#   del st.session_state.context
# st.rerun()



st.success("all PDF Processed")

if "chat" not in st.session_state:
  st.session_state.chat = []

query =st.chat_input("Ask something...")

  # if "eval_data" not in st.session_state:
  #   st.session_state.eval_data =[]

if "eval_data" not in st.session_state:
  st.session_state.eval_data =[]



if query:
  answer, context = ask_question(query,embeddings,chunks)

  st.session_state.chat.append(("user",query))
  st.session_state.chat.append(("bot",answer))
  st.session_state.context = context



  st.session_state.eval_data.append({
      "question": str(query) if query else " ",
      "answer": str(answer) if answer else " ",
      "contexts": [str(context)] if context else [""],
      "ground_truth": str(context) if context else " "
    })

for role,msg in st.session_state.chat:
  with st.chat_message(role):
    st.write(msg)

if "context" in st.session_state:
  with st.expander ("context used"):
    st.write(st.session_state.context)

if st.button("run evaluation"):

  if "eval_data" in st.session_state and len(st.session_state.eval_data) > 0:
    st.write("running evaluation")



      # dataset =Dataset.from_list(st.session_state.eval_data)

    result, df = run_evaluate(st.session_state.eval_data)
    st.write(df)


  else:
    st.warning("no data to evaluate")


  # if st.button("evaluate RAG"):
  #   if "eval_data" in st.session_state and st.session_state.eval_data:

  #     dataset = Dataset.from_list(st.session_state.eval_data)
  #     dataset = evaluate(
  #         dataset,
  #         metrics=[
  #             faithfulness,
  #             answer_relevancy,
  #             context_precision,
  #             context_recall
  #         ]

  #     )

  #     st.write("Evaluation Results")
  #     st.write(result)

#     else:
#       st.writing("No data to evaluate")

# import json

# with open("eval_data.json", "w") as f:
#   json.dump(st.session_state.eval_data,f)





