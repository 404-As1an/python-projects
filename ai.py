# -----------------------------------------------------------
# local_gemma_pipeline.py
# Example: Running a local Hugging Face model (Gemma-3-1b-it)
# -----------------------------------------------------------

# Import the required modules from Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch

# Path to your locally downloaded model folder
# Make sure this folder contains model files (config.json, model.safetensors, etc.)
MODEL_PATH = "./models/gemma-3-1b-it"

# -----------------------------------------------------------
# STEP 1: Load the tokenizer
# The tokenizer converts human-readable text into model-readable tokens.
# This must match the exact model configuration.
# -----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -----------------------------------------------------------
# STEP 2: Load the model
# The AutoModelForCausalLM class loads a model designed for text generation (causal language modeling).
# 'torch_dtype' and 'device_map' are performance options:
#  - torch_dtype=torch.bfloat16: use bfloat16 precision (if supported by GPU)
#  - device_map="auto": automatically assign model layers to GPU(s) if available
# -----------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # Updated argument name
    device_map="auto" if torch.cuda.is_available() else None               # Let Accelerate manage devices
)

# Create a streamer to print tokens in real time
streamer = TextStreamer(tokenizer)

# -----------------------------------------------------------
# STEP 3: Create a text generation pipeline
# The 'pipeline' function simplifies inference by combining model + tokenizer + generation config.
# -----------------------------------------------------------
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# -----------------------------------------------------------
# STEP 4: Generate text from a prompt
# 'max_new_tokens' controls how many tokens (words) to generate.
# 'temperature' controls randomness (higher = more creative).
# -----------------------------------------------------------
prompt = "Explain in detail dimensions"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Run the model pipeline
# Generate with streaming output
output_tokens = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode tensor -> text
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# -----------------------------------------------------------
# STEP 5: Print the generated text
# The result is a list of dicts: [{'generated_text': '...'}]
# -----------------------------------------------------------
# Print and save
print("\nModel Output:\n")
print(output_text)
# print(result[0]['generated_text'])
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
print("\n\n✅ Full output saved to output.txt")