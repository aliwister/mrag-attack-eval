# Define the mapping
llm_mapper = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen-VL-7B",
    "nvidia/Cosmos-Reason1-7B": "Cosmos-7B",
    "OpenGVLab/InternVL3_5-8B-HF": "InternVL-8B",
    "google/gemma-3-12b-it": "Gemma-12B",
    "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-Mistral-7B",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama-11B-V"
}

# Simple helper function
def map_llm_name(name: str) -> str:
    return llm_mapper.get(name, name) 