from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__, static_folder='static')
app.secret_key = "supersecret"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY not found. Using dummy values.")

if not GEMINI_API_KEY:
    raise ValueError(" Missing GEMINI_API_KEY in .env file")

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
print(f"Connecting to Pinecone index: {index_name}...")
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
except Exception as e:
    print(f"Error connecting to Pinecone or index not found: {e}")
    print("Application will run with a placeholder retriever, but RAG will fail.")
    from langchain.retrievers import EnforcedContextRetriever
    class DummyRetriever:
        def get_relevant_documents(self, query):
            return []
        def as_retriever(self, **kwargs):
            return self

    docsearch = DummyRetriever()

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
)

def get_chat_history():
    history = session.get("chat_history", [])
    messages = []
    for m in history:
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "ai": 
            messages.append(AIMessage(content=m.get("content", "")))
    return messages

def save_message(user_msg, bot_msg):
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": user_msg})
    session["chat_history"].append({"role": "ai", "content": bot_msg}) 
    session.modified = True


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be used to retrieve relevant documents from a vector store. The standalone question should summarize the context if necessary."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(
    chatModel, 
    retriever, 
    contextualize_q_prompt
)

question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


@app.route("/")
def index():
    session.pop("chat_history", None)
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg:
        return "Please enter a message.", 400

    history_messages = get_chat_history()

    try:
        response = rag_chain.invoke({
            "input": msg, 
            "chat_history": history_messages
        })
        answer = response["answer"]
        
        save_message(msg, answer)

        return str(answer)

    except Exception as e:
        print(f"An error occurred during RAG chain invocation: {e}")
        return "Sorry, I am currently unable to process your request. Please check the server logs for API errors.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
