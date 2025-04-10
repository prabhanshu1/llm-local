from flask import Flask, request, jsonify
import torch
import time
from transformers import AutoTokenizer, Gemma3ForCausalLM
import prompts

app = Flask(__name__)

# Load model and tokenizer
ckpt = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(
    ckpt, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(ckpt)






@app.route("/api/chat/", methods=["POST"])
def chat():
    start_time = time.time()
    user_msg = request.json.get("messages", "")
    if not user_msg:
        return jsonify({"error": "Message is required"}), 400

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompts.LLM_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}],
            },
        ]
    ]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
    end = time.time()
    print(f"Execution time: {end - start_time:.3f} seconds")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
