"""
Program 26: Text Generation using GPT-2 on a custom dataset
Fine-tunes GPT-2 on data/custom_corpus.txt, then generates text from prompts.
"""
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

with open("data/custom_corpus.txt", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

enc = tok(lines, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
data = list(zip(enc["input_ids"], enc["attention_mask"]))
loader = DataLoader(data, batch_size=4, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    total = 0.0
    for ids, mask in loader:
        ids, mask = ids.to(device), mask.to(device)
        labels = ids.clone()
        labels[mask == 0] = -100                  # ignore padding in loss
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        total += loss.item()
    print(f"Epoch {epoch + 1}: loss = {total / len(loader):.4f}")

model.eval()
for prompt in ["Natural language processing", "Machine learning models", "Attention mechanisms"]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=40, do_sample=True, top_k=50, top_p=0.95,
                         temperature=0.8, pad_token_id=tok.eos_token_id)
    print(f"\nPrompt: {prompt}\n-> {tok.decode(out[0], skip_special_tokens=True)}")
