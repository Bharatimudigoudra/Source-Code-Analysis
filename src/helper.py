import os
from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
#from langchain_community.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#clone any github repositories 
def repo_ingestion(repo_url):
    repo_path = "repo"  # Define the directory where you clone the repo

    # Check if the directory already exists
    if os.path.exists(repo_path):
        print(f"Directory '{repo_path}' already exists. Deleting and re-cloning...")
        shutil.rmtree(repo_path)  # Delete the existing directory

    # Clone the repository
    Repo.clone_from(repo_url, to_path=repo_path)




#Loading repositories as documents
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
                                        )
    
    documents = loader.load()

    return documents




#Creating text chunks 
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language = Language.PYTHON,
                                                             chunk_size = 2000,
                                                             chunk_overlap = 200)
    
    text_chunks = documents_splitter.split_documents(documents)

    return text_chunks



#loading embeddings model
def load_embedding():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
