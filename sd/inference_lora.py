import torch
from PIL import Image
from transformers import CLIPTokenizer
import numpy as np

from sd.model_loader import preload_models_from_standard_weights
from sd.pipeline import generate
from lora import inject_lora_to_diffusion_unet, load_lora_weights

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/content/Stable-Diffusion-from-scratch/data/v1-5-pruned-emaonly.ckpt"
VOCAB_PATH = "/content/Stable-Diffusion-from-scratch/data/vocab.json"
MERGES_PATH = "/content/Stable-Diffusion-from-scratch/data/merges.txt"

# LoRA Configuration
LORA_RANK = 4
LORA_ALPHA = 1.0
LORA_WEIGHTS_PATH = "/content/checkpoints_lora/lora_best.pt"  # or lora_final.pt

def load_model_with_lora(model_path, lora_path, device):
    """Load base model and inject LoRA weights"""
    print("Loading base model...")
    models = preload_models_from_standard_weights(model_path, device)
    
    print("Injecting LoRA layers...")
    diffusion = models['diffusion']
    diffusion, _ = inject_lora_to_diffusion_unet(
        diffusion,
        rank=LORA_RANK,
        alpha=LORA_ALPHA
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    diffusion = load_lora_weights(diffusion, lora_path)
    
    models['diffusion'] = diffusion
    return models

def generate_image(
    prompt,
    models,
    tokenizer,
    uncond_prompt="",
    seed=42,
    cfg_scale=7.5,
    num_inference_steps=50,
    device="cuda"
):
    """Generate image using the model with LoRA"""
    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    
    return Image.fromarray(output_image)

def main():
    print(f"Using device: {DEVICE}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)
    
    # Load model with LoRA
    models = load_model_with_lora(MODEL_PATH, LORA_WEIGHTS_PATH, DEVICE)
    
    print("\n" + "="*60)
    print("Model loaded successfully with LoRA weights!")
    print("="*60)
    
    # Example prompts
    prompts = [
        "Naruto Uzumaki in sage mode, highly detailed, ultra sharp, cinematic, 8k resolution",
        "Sasuke Uchiha with sharingan eyes, dramatic lighting, highly detailed, 8k resolution",
        "Sakura Haruno portrait, beautiful anime style, highly detailed, cinematic lighting",
    ]
    
    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}/{len(prompts)}...")
        print(f"Prompt: {prompt}")
        
        image = generate_image(
            prompt=prompt,
            models=models,
            tokenizer=tokenizer,
            uncond_prompt="blurry, low quality, distorted",
            seed=42 + i,
            cfg_scale=7.5,
            num_inference_steps=50,
            device=DEVICE
        )
        
        # Save image
        output_path = f"output_lora_{i+1}.png"
        image.save(output_path)
        print(f"Saved: {output_path}")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()