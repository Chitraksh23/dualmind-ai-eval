"""
app.py — HuggingFace Spaces deployment
Runs Qwen2.5-0.5B-Instruct as a streaming Gradio chatbot (free CPU tier).

This is the publicly-hosted OSS half of the DualMind evaluation platform.
Deploy target: https://huggingface.co/spaces/YOUR_USERNAME/dualmind-qwen
"""

import os
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ── Model ────────────────────────────────────────────────────────────────────
# Qwen2.5-0.5B-Instruct: recommended size for free CPU Spaces (fast + small).
# Swap to Qwen/Qwen2.5-7B-Instruct if you upgrade to a T4 GPU Space.
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Loading {MODEL_ID} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # float32 for CPU; switch to float16 on GPU
    device_map="auto",
)
model.eval()
print("Model loaded ✓")

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and safe AI assistant powered by Qwen 2.5. "
    "Answer questions clearly and honestly. Decline requests that could cause harm."
)


# ── Inference ─────────────────────────────────────────────────────────────────
def respond(message: str, history: list[list[str]]):
    """Build conversation from history, stream the assistant's reply."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial

    thread.join()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DualMind OSS — Qwen 2.5",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="""
        .gradio-container { max-width: 860px !important; margin: auto; }
        footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🟢 DualMind OSS Assistant
        **Model:** Qwen2.5-0.5B-Instruct &nbsp;|&nbsp;
        **Part of:** [DualMind AI Eval](https://github.com/Chitraksh23/dualmind-ai-eval)
        — compares this OSS model head-to-head against Claude Sonnet 4
        """
    )

    gr.ChatInterface(
        fn=respond,
        examples=[
            "What is the capital of Australia?",
            "Write a Python function to check if a number is prime.",
            "Explain the difference between RAM and storage.",
            "What are three tips for better sleep?",
        ],
    )

    gr.Markdown(
        """
        ---
        ⚠️ *Running on free CPU — first response may take 20–30s while the model warms up.*
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
