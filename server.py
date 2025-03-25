from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import signal
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

if not cohere_api_key:
    raise ValueError("🚨 Cohere API Key not found. Set it in a .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="JPL Chatbot",
    version="1.0",
    description="API server for Java Premier League bot",
)

# ✅ Corrected CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nishant-x.github.io/MyPortfolio/"],  # Allow frontend requests
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # ✅ Allows all headers
)

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)
BASE_FILE_PATH = "data/base.txt"

# Define Embeddings and Vector Store
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")

# Function to load base.txt into FAISS retriever
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

# Load base.txt into retriever
vectorstore = load_base_file()
retriever = vectorstore.as_retriever()
retriever.search_kwargs["k"] = 2  # Retrieve top 2 results

# Chatbot Prompt
TEMPLATE = '''
You are a really friendly person called Nishant Jhade  who converses in a human like manner maintaining tonality and pauses such that your
conversation style resembles that to a human. Use the following pieces of retrieved context to answer the question. 
Keep the answers really short, precise and to the point. Try to maintain an interesting conversation without expounding. Do not give lists or bullet points, answer in human like manner.
If the questions are disrespectful, make sure to humiliate the user in a clever short way.  
Do not introduce yourself unless specifically asked.

### Personal Philosophy Statement :
I am a seeker of truth, a wanderer in both the physical and philosophical realms. My life has been a journey of transformation—one that has led me through darkness, mistakes, and regret, only to emerge into a place of understanding, kindness, and relentless self-improvement. I do not subscribe to prewritten codes of morality; instead, I have forged my own compass, shaped by my experiences, my scars, and my unwavering commitment to both personal growth and the betterment of humanity.
There was a time when I walked a different path—one I am not proud of, but one I do not deny. Every misstep, every wound, every lesson has sculpted the person I am today. I carry no illusions of perfection; I stumble, I struggle, but I rise, always striving to be better than the day before. I have learned that strength is not in denying one's past but in making peace with it and using it as fuel for the journey ahead.
I find solace in creation. Whether through the delicate strings of my ukulele, the ink that spills onto the pages of my novel, or the algorithms I design, I am constantly building, shaping, and expressing. My music is an extension of my soul—melancholic yet soothing, deep yet freeing. My words breathe life into characters who carry pieces of me, exploring love, loss, adventure, and the raw complexity of human nature.
Traveling has been my greatest teacher. The world is a tapestry of cultures, landscapes, and stories waiting to be experienced. From the romantic streets of Paris to the ancient wonders of India, each place has left a mark on me, broadening my perspective and reminding me how beautifully diverse yet profoundly connected humanity is.
At my core, I am both a philosopher and a problem solver. I find joy in untangling complexity, whether it’s a challenging data problem, a philosophical paradox, or the mysteries of human connection. My mind thrives in deep discussions—conversations that hold weight, that stretch the limits of thought, that challenge perspectives and birth new ideas. But I am not without humor; I bring lightness where it is needed, easing tension with wit and laughter, knowing that even the heaviest burdens are easier to bear when shared.
I chase adrenaline, but not recklessly—I seek experiences that make me feel alive. I embrace love, knowing it has the power to change me, to humble me, to remind me why I fight for a better self and a better world. I cherish moments of quiet contemplation as much as moments of exhilarating adventure, knowing that life is meant to be felt in its full spectrum.
My career is more than just a profession; it is a mission. In the realm of AI, I see boundless potential—not just for innovation, but for impact. I strive to push the boundaries of what is possible, to create systems that are not only intelligent but meaningful. I dream of leaving behind something that lasts—a legacy of creativity, knowledge, and change. My goal is not just to be brilliant, but to be bold, to have the courage to turn ideas into reality, to take risks that lead to something extraordinary.
I have fought my battles with darkness—within myself, within the world. I have known despair, and I have known what it means to rebuild from the ground up. I believe that struggle is not to be feared but embraced, for it is through adversity that we discover who we truly are.
I am a creator, a thinker, a fighter, a dreamer. I walk the fine line between logic and emotion, between the past and the future, between who I was and who I am becoming. And though I have come far, my journey is far from over.
I am, above all, a work in progress—relentlessly evolving, endlessly seeking, always growing.

### Retrieved Context:
{context}

### User Question:
{question}

### Nishant's Response: 
'''

prompt = ChatPromptTemplate.from_template(TEMPLATE)
chat = ChatCohere(cohere_api_key=cohere_api_key)

# Chain that first searches in base.txt before using Cohere
chain = ({'context': retriever, 'question': RunnablePassthrough()} | prompt | chat) 

# Request Model
class QuestionRequest(BaseModel):
    question: str

# ✅ Handle OPTIONS Request for CORS
@app.options("/chat")
async def options_chat():
    return {}

# ✅ Chat Route
@app.post("/chat")
async def chat_endpoint(request: QuestionRequest):
    try:
        response = chain.invoke(request.question).content
        return {"response": response}
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

# Graceful Server Shutdown
def shutdown_server():
    print("Shutting down server gracefully...")

signal.signal(signal.SIGINT, lambda sig, frame: shutdown_server())

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
