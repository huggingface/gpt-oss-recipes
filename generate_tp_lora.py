import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# --- Configuration ---
# Path to your saved PEFT adapter (after you finetuned)
adapter_path = "/fsx/ferdinandmom/ferdinand-hf/gpt-oss-recipes/gpt-oss-20b-multilingual-reasoner"

# 1. Load the base model path from the adapter's config file
with open(f"{adapter_path}/adapter_config.json", "r") as f:
    adapter_config = json.load(f)
base_model_path = adapter_config["base_model_name_or_path"]

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path, padding_side="left")

# 3. Prepare inputs
messages = [{"role": "user", "content": "Explain tensor parallelism in simple terms."}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
)

# 4. Set up tensor parallelism
device_map = {"tp_plan": "auto"}

# 5. Load the base model and apply tensor parallelism FIRST
print(f"Loading base model `{base_model_path}` with tensor parallelism...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    **device_map,
)

# 6. Load the LoRA adapter onto the sharded base model
print(f"Loading adapter `{adapter_path}` onto the sharded model...")
model = PeftModel.from_pretrained(base_model, adapter_path)

model.eval()

# 7. Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)

# 8. Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response.split("assistant\n")[-1].strip())