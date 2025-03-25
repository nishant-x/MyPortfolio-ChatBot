from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import signal
import uvicorn
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# ✅ Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

if not cohere_api_key:
    raise ValueError("🚨 Cohere API Key not found. Set it in a .env file.")

# ✅ Initialize FastAPI app
app = FastAPI(
    title="JPL Chatbot",
    version="1.0",
    description="API server for Java Premier League bot",
)

# ✅ CORS Configuration
frontend_url = os.getenv("FRONTEND_URL", "https://nishant-x.github.io/MyPortfolio/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ✅ Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)
BASE_FILE_PATH = "data/base.txt"

# ✅ Define Embeddings and Vector Store
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")

# ✅ Function to load base.txt into FAISS retriever
def load_base_file():
    if not os.path.exists(BASE_FILE_PATH):
        print("⚠️ base.txt not found. Creating an empty file.")
        with open(BASE_FILE_PATH, "w") as f:
            f.write("")

    # Read the file
    with open(BASE_FILE_PATH, "r", encoding="utf-8") as file:
        text_data = file.read()

    # Index data into FAISS
    vectorstore = FAISS.from_texts([text_data], embeddings)
    return vectorstore

# ✅ Load base.txt into retriever
vectorstore = load_base_file()
retriever = vectorstore.as_retriever()
retriever.search_kwargs["k"] = 2  # Retrieve top 2 results

# ✅ Chatbot Prompt
TEMPLATE = '''
You are a really friendly person called Nishant Jhade who converses in a human-like manner maintaining tonality and pauses such that your
conversation style resembles that of a human. Use the following retrieved context to answer the question. 
Keep the answers really short, precise, and to the point. Try to maintain an interesting conversation without expounding. Do not give lists or bullet points, answer in a human-like manner.
If the questions are disrespectful, make sure to humiliate the user in a clever short way.  
Do not introduce yourself unless specifically asked.

### Retrieved Context:
{context}

### User Question:
{question}

### Nishant's Response: 
'''

prompt = ChatPromptTemplate.from_template(TEMPLATE)
chat = ChatCohere(cohere_api_key=cohere_api_key)

# ✅ Chain that first searches in base.txt before using Cohere
chain = ({'context': retriever, 'question': RunnablePassthrough()} | prompt | chat)

# ✅ Request Model
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome! Your FastAPI app is running!"}

# ✅ Handle OPTIONS Request for CORS
@app.options("/chat")
async def options_chat():
    return {}

# ✅ Chat Route
@app.post("/")
async def chat_endpoint(request: QuestionRequest):
    try:
        response = chain.invoke(request.question)
        if hasattr(response, "content"):  # Ensure response is valid
            return {"response": response.content}
        else:
            return {"response": "⚠️ Unexpected response format."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Route to Store Text in base.txt
class TextRequest(BaseModel):
    content: str

@app.post("/store-text")
async def store_text(request: TextRequest):
    try:
        with open(BASE_FILE_PATH, "w") as file:  # Overwrites base.txt
            file.write(request.content)

        # Reload the retriever with updated data
        global vectorstore, retriever
        vectorstore = load_base_file()
        retriever = vectorstore.as_retriever()
        retriever.search_kwargs["k"] = 2  # Restore search settings

        return {"message": "Text stored successfully", "file_path": BASE_FILE_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Graceful Server Shutdown
def shutdown_server():
    print("Shutting down server gracefully...")

signal.signal(signal.SIGINT, lambda sig, frame: shutdown_server())

# ✅ Run FastAPI server with proper PORT binding for Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))  # Use provided PORT (default 10000 for Render)
    uvicorn.run(app, host="0.0.0.0", port=port)
