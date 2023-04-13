%pip install accelerate>=0.12.0 transformers[torch]==4.25.1

import torch
from transformers import pipeline

# original
# generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")


# per the post https://huggingface.co/databricks/dolly-v2-12b , using local copy of instruct_pipeline.py instead
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto")

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
