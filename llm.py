from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import time

app = FastAPI()

# Load the quantized Gemma 3 4B model directly from Hugging Face
llm = Llama.from_pretrained(
    repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
    filename="gemma-3-4b-it-q4_0.gguf",
    n_ctx=4096,
    n_gpu_layers=100,
    n_threads=4,
    n_batch=64,
    verbose=False
)

_ = llm("Warm-up prompt", max_tokens=1)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: list[Message]
    temperature: float = 0.7

@app.post("/api/chat/")
async def chat(request: ChatRequest):
    start = time.time()

    # Convert messages into plain text prompt
    conversation = ""
    for msg in request.message:
        prefix = "User" if msg.role == "user" else "Assistant"
        conversation += f"{prefix}: {msg.content.strip()}\n"
    conversation += "Assistant:"

    # Generate
    output = llm(
        conversation,
        max_tokens=64,
        temperature=request.temperature,
        stop=["User:", "Assistant:"],
        stream=True
    )

    response_text = output["choices"][0]["text"].strip()
    print(f"⏱️ Latency: {time.time() - start:.2f}s")

    return {"response": response_text}
