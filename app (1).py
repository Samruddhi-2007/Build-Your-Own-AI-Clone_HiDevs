import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

#  NEW imports for HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

#  API keys (if needed elsewhere)
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxx"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxx"

st.title("🤖 Your AI Chat Clone")

loader = PyPDFLoader("Syllabus-Universal-Human-Values.pdf")  # Update file if needed
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

# Load Hugging Face Model via Pipeline (REPLACEMENT)
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit interface
query = st.text_input("Ask me anything from your data:")
if query:
    response = qa.run(query)
    st.write(response)
