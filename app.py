from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.secret_key = "supersecret"   

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(" Missing PINECONE_API_KEY in .env file")

if not GEMINI_API_KEY:
    raise ValueError(" Missing GEMINI_API_KEY in .env file")

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY
)

# ------------------- Chat History ------------------- #
def get_chat_history():
    """Retrieve chat history from Flask session"""
    history = session.get("chat_history", [])
    messages = []
    for m in history:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") in ["ai", "bot"]:  # normalize both to AIMessage
            messages.append(AIMessage(content=m.get("content", "")))
    return messages

def save_message(user_msg, bot_msg):
    """Save user + bot message to Flask session"""
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": user_msg})
    session["chat_history"].append({"role": "ai", "content": bot_msg})  # 👈 use 'ai' not 'bot'
    session.modified = True


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Create chains
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
history_aware_retriever = create_history_aware_retriever(chatModel, retriever, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ---------------- Routes ---------------- #
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Load session chat history
    history = get_chat_history()

    # RAG call with history
    response = rag_chain.invoke({"input": msg, "history": history})
    answer = response["answer"]

    # Save both user + bot messages
    save_message(msg, answer)

    return str(answer)

# ---------------- Run Server ---------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
