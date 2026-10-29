"""
Program 30: Chatbot Development using a pre-trained transformer (DialoGPT)
Usage:
  python dialogpt_chatbot.py          # interactive chat (type 'quit' to exit)
  python dialogpt_chatbot.py --demo   # scripted demo conversation
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/DialoGPT-small"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()


def chat(user_inputs):
    history = None
    for text in user_inputs:
        new_ids = tok.encode(text + tok.eos_token, return_tensors="pt")
        input_ids = new_ids if history is None else torch.cat([history, new_ids], dim=-1)
        input_ids = input_ids[:, -300:]                     # keep context short
        with torch.no_grad():
            history = model.generate(input_ids, max_new_tokens=60, do_sample=True, top_k=50,
                                     top_p=0.92, temperature=0.8,
                                     pad_token_id=tok.eos_token_id,
                                     attention_mask=torch.ones_like(input_ids))
        reply = tok.decode(history[0, input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"You: {text}\nBot: {reply}\n")


if "--demo" in sys.argv:
    chat(["Hello, how are you?", "What do you like to do?", "Tell me something interesting."])
else:
    history = None
    print("DialoGPT chatbot (type 'quit' to exit)")
    while True:
        text = input("You: ")
        if text.strip().lower() in {"quit", "exit"}:
            break
        new_ids = tok.encode(text + tok.eos_token, return_tensors="pt")
        input_ids = new_ids if history is None else torch.cat([history, new_ids], dim=-1)
        input_ids = input_ids[:, -300:]
        with torch.no_grad():
            history = model.generate(input_ids, max_new_tokens=60, do_sample=True, top_k=50,
                                     top_p=0.92, temperature=0.8,
                                     pad_token_id=tok.eos_token_id,
                                     attention_mask=torch.ones_like(input_ids))
        print("Bot:", tok.decode(history[0, input_ids.shape[-1]:], skip_special_tokens=True))
