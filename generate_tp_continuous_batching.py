import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Model configuration - uncomment the model size you want to use
model_path = "openai/gpt-oss-120b"  # 120B model (default)
# model_path = "openai/gpt-oss-20b"  # 20B model - uncomment this line and comment the line above

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

# Create a batch of different prompts
batch_messages = [
    [
        {
            "role": "user",
            "content": "Explain how expert parallelism works in large language models.",
        }
    ],
    [
        {
            "role": "user",
            "content": "What are the advantages of tensor parallelism over data parallelism?",
        }
    ],
    [
        {
            "role": "user",
            "content": "How does continuous batching improve inference throughput in LLMs?",
        }
    ],
    [
        {
            "role": "user",
            "content": "Compare the memory requirements of different parallelism strategies.",
        }
    ],
    [
        {
            "role": "user",
            "content": "What role does attention mechanism play in transformer models?",
        }
    ],
]

# Apply chat template to each set of messages
chat_prompts = [
    tokenizer.apply_chat_template(messages, tokenize=False)
    for messages in batch_messages
]

generation_config = GenerationConfig(
    max_new_tokens=1000,
)

device_map = (
    {
        "tp_plan": "auto",  # Enable Tensor Parallelism
    }
    if "120b" in model_path
    else {"device_map": "auto"}
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    use_kernels=True,
    attn_implementation="paged_attention|kernels-community/vllm-flash-attn3",
    **device_map,
)

model.eval()

# Tokenize the batch of prompts
inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to("cuda")

print(f"Processing batch of {len(chat_prompts)} prompts...")
print("=" * 80)

# Generate responses for all prompts in the batch
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print all responses
for i, output in enumerate(outputs):
    response = tokenizer.decode(output, skip_special_tokens=True)
    # Extract just the assistant's response part
    assistant_response = response.split("assistant\n")[-1].strip()

    print(f"Prompt {i + 1}: {batch_messages[i][0]['content'][:50]}...")
    print(f"Response {i + 1}: {assistant_response}")
    print("-" * 80)
