
# app.py
import os
import threading
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
SYSTEM_PROMPT = (
    "You are Yomex AI, a concise, friendly, and helpful assistant. "
    "Be accurate, avoid making things up, and keep answers short unless the user asks for detail."
)

# -------- Load model & tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # safety for generation

# Use GPU if available; otherwise CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",           # GPU if available, else CPU
    torch_dtype=torch.float32    # use float32 on CPU; HF Spaces CPU is fine
)
DEVICE = model.device

# -------- Helpers --------
def build_messages(history, user_input: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in (history or []):
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})
    return messages

# -------- Streaming generator for Gradio --------
def response_stream(user_input: str, history: list):
    if not user_input or not user_input.strip():
        yield history, ""  # no change, just clear box
        return

    # Keep context short (last N turns) to stay fast on CPU Spaces
    MAX_TURNS = 8
    if history and len(history) > MAX_TURNS:
        history = history[-MAX_TURNS:]

    messages = build_messages(history, user_input)

    # Build prompt and tokenize
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Set up a streamer that yields tokens as they are generated
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Background generation so we can iterate the streamer immediately
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=256,         # tune for speed
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens into the chat bubble
    partial = ""
    # Show an empty assistant bubble immediately (nice UX)
    running_history = history + [[user_input, ""]]
    yield running_history, ""  # clears the textbox right away

    for token in streamer:
        partial += token
        running_history[-1][1] = partial
        yield running_history, ""  # live update

# -------- Clear handler --------
def clear_chat():
    return [], ""

# -------- UI --------
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("<h1 style='text-align:center;'>🤖 Yomex AI — Phi-3 Mini (Streaming)</h1>")
    chatbot = gr.Chatbot(
        height=480,
        bubble_full_width=False,
        show_copy_button=True,
        avatar_images=(None, None)
    )
    with gr.Row():
        user_in = gr.Textbox(
            placeholder="Type your message and press Enter…",
            scale=8
        )
        clear_btn = gr.Button("Clear", scale=1)

    # Stream directly: inputs = (message, current history), outputs = (updated history, cleared textbox)
    user_in.submit(response_stream, [user_in, chatbot], [chatbot, user_in], queue=True)
    clear_btn.click(clear_chat, outputs=[chatbot])

if __name__ == "__main__":
    #demo.launch()
    