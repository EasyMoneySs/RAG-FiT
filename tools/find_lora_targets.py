import argparse
import sys
from transformers import AutoModelForCausalLM

def find_target_modules(model_name):
    print(f"Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    print("\nModel Architecture:")
    print(model)
    
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear,)):
             # Extract the last part of the name (e.g., 'q_proj' from 'model.layers.0.self_attn.q_proj')
            layer_name = name.split('.')[-1]
            linear_layers.add(layer_name)
        # Also check for Conv1D which is sometimes used in older models (like GPT-2) instead of Linear
        # But for modern LLMs Linear is standard.

    print("\nPotential target_modules for LoRA (Linear layers found):")
    for layer in sorted(list(linear_layers)):
        print(f"  - {layer}")

    print("\nCommon LoRA target suggestions:")
    suggested = []
    # Heuristics for common attention/mlp projections
    common_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'query_key_value']
    for layer in linear_layers:
        if any(p in layer for p in common_patterns):
            suggested.append(layer)
    
    if suggested:
        print(f"  target_modules: {suggested}")
    else:
        print("  (No standard attention/MLP layer names matched. Please inspect the architecture above manually.)")

if __name__ == "__main__":
    import torch # Imported here to avoid check if script is just printing help
    
    parser = argparse.ArgumentParser(description="Find potential target_modules for LoRA configuration.")
    parser.add_argument("model_name", type=str, help="Hugging Face model name or path (e.g., 'microsoft/Phi-3-mini-128k-instruct')")
    
    args = parser.parse_args()
    
    find_target_modules(args.model_name)
