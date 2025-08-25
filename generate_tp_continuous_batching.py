import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Model configuration - uncomment the model size you want to use
model_path = "openai/gpt-oss-120b"  # 120B model (default)
# model_path = "openai/gpt-oss-20b"  # 20B model - uncomment this line and comment the line above

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

# Ensure tokenizer has a pad token (required for batch processing)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# Apply chat template to each set of messages and tokenize in one step
def tokenize_function(messages):
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(chat_prompt)

# Process all messages and get input_ids
batch_input_ids = [
    tokenize_function(messages)["input_ids"] 
    for messages in batch_messages
]

generation_config = GenerationConfig(
    max_new_tokens=1000,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,  # Use greedy decoding by default
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

# batch_input_ids is already prepared above with chat template applied

print(f"Processing batch of {len(batch_input_ids)} prompts...")
print("=" * 80)

# Generate responses for all prompts in the batch using continuous batching
batch_outputs = model.generate_batch(
    inputs=batch_input_ids,
    generation_config=generation_config,
)

# Decode and print all responses
for i, request_id in enumerate(batch_outputs):
    request_output = batch_outputs[request_id]
    
    # Decode the prompt
    input_text = tokenizer.decode(request_output.prompt_ids, skip_special_tokens=True)
    
    # Decode the generated tokens
    try:
        output_text = tokenizer.decode(request_output.generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Decoding failed for request {request_id}: {e}")
        output_text = tokenizer.decode(request_output.generated_tokens[1:], skip_special_tokens=True)
    
    # Extract just the assistant's response part
    assistant_response = output_text.split("assistant\n")[-1].strip() if "assistant\n" in output_text else output_text.strip()

    print(f"Prompt {i + 1}: {batch_messages[i][0]['content'][:50]}...")
    print(f"Response {i + 1}: {assistant_response}")
    print("-" * 80)
