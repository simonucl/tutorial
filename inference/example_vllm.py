from vllm import LLM, SamplingParams
import datasets
from transformers import AutoTokenizer

model="meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

# Sample prompts.
alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
prompts = []
for example in alpaca_eval_data:
    prompt = example["instruction"]
    prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                 tokenize=False, 
                                                 add_generation_prompt=True))

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

# Create an LLM.
llm = LLM(model=model, trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
generation = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generation.append(generated_text)

print(generation[0])