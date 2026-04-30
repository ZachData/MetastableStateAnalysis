from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("albert-base-v2")
import json
prompts = json.load(open("p2c_churchland/prompt_grids/matched_length.json"))
for p in prompts:
    print(len(tok(p["prompt"])["input_ids"]), p["prompt"][:40])