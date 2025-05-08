import json
from pcap_tokenization2 import generate_variations

JSON_PATH = "streams.json"
OUTPUT_TXT_FILE = "all_prompts.txt"

all_prompts = []

with open(JSON_PATH, 'r') as f:
    streams = json.load(f)
    
for stream in streams:
    base_prompt = stream["prompt"]
    stream_id = stream.get("stream_id", {})
    stream_id_info = " ".join(f"{key}: {value}" for key, value in stream_id.items())
    prompts = generate_variations(base_prompt, stream_id_info)
    
    for p in prompts:
        all_prompts.append(p)
    
    
with open(OUTPUT_TXT_FILE, 'w') as f:
    for prompt in all_prompts:
        f.write(prompt.strip() + "\n")