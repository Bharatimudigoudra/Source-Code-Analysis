from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import os


load_dotenv()

Groq_API_KEY = os.environ.get('Groq_API_KEY')
os.environ["Groq_API_KEY"] = Groq_API_KEY

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict




# url = "https://github.com/Bharatimudigoudra/Medical-Chatbot-GenAI"

# repo_ingestion(url)


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()



#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()