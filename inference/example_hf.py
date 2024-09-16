from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
from tqdm import trange

batch_size = 16
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")

# Sample prompts.
alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
prompts = []
for example in alpaca_eval_data:
    prompt = example["instruction"]
    prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                 tokenize=False, 
                                                 add_generation_prompt=True))

# Generate texts from the prompts.
generation = []
for i in trange(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generation.extend(generated_texts)

print(generation[0])
