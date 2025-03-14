from langchain_chroma import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)


load_dotenv()

Groq_API_KEY = os.environ.get('Groq_API_KEY')
os.environ["Groq_API_KEY"] = Groq_API_KEY



embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)
memory = ConversationBufferMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


