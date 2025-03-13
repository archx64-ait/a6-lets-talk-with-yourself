import torch
import os
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
import json

app = Flask(__name__, template_folder='templates')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load FAISS index using the same embeddings as the notebook
embedding_model = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vector_db = FAISS.load_local('vector-store/faiss_index', embeddings, allow_dangerous_deserialization=True)

# Load TinyLlama model
model_name = 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
model_path = os.path.join('models', model_name)

if not os.path.exists(model_path):
    print(f'Error: Model file {model_name} not found!')
    llm = None
else:
    llm = Llama(model_path=model_path, n_gpu_layers=30, n_threads=8, n_ctx=1024)

CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except json.JSONDecodeError:
            print("Warning: chat_history.json is corrupted. Resetting history.")
            return []
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

chat_history = load_chat_history()


@app.route('/')
def index():
    return render_template('chat.html')


def rag_pipeline(query):
    if vector_db is None:
        return 'Error: No vector database available.', []
    
    retrieved_docs = vector_db.similarity_search(query, k=4)
    context = '\n'.join([doc.page_content for doc in retrieved_docs])
    sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
    
    prompt = f'''
    You are an AI assistant designed to answer questions about Kaung SiThu. 
    Be gentle, informative, and concise. If you don't know an answer, politely say no.
    
    Context:
    {context}
    
    User Question: {query}
    '''
    
    response = llm(prompt)
    return response['choices'][0]['text'].strip(), sources


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    if not query:
        return jsonify({'error': 'No question provided'}), 400
    
    answer, sources = rag_pipeline(query)
    
    chat_entry = {"question": query, "answer": answer, "sources": sources}
    chat_history.append(chat_entry)
    save_chat_history(chat_history)
    
    return jsonify(chat_entry)


chat_history = []

@app.route('/history', methods=['GET'])
def history():
    return jsonify(chat_history)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
