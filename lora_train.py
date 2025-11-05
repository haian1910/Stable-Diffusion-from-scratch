import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import gc

# Import your SD modules
from sd.model_loader import preload_models_from_standard_weights
from sd.ddpm import DDPMSampler
from lora import inject_lora_to_diffusion_unet, get_lora_parameters, save_lora_weights

# Configuration
BATCH_SIZE = 1
LEARNING_RATE = 1e-4  # Higher LR for LoRA
NUM_EPOCHS = 10
SAVE_EVERY = 50  # Save checkpoint every N steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# LoRA Configuration
LORA_RANK = 4  # Rank of LoRA matrices (4, 8, 16, 32)
LORA_ALPHA = 1.0  # Scaling factor
USE_GRADIENT_CHECKPOINTING = False  # Can keep False with LoRA

# Paths
MODEL_PATH = "/content/Stable-Diffusion-from-scratch/data/v1-5-pruned-emaonly.ckpt"
VOCAB_PATH = "/content/Stable-Diffusion-from-scratch/data/vocab.json"
MERGES_PATH = "/content/Stable-Diffusion-from-scratch/data/merges.txt"
OUTPUT_DIR = "./checkpoints_lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()

def rescale(x, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    return x

def preprocess_image(image):
    """Preprocess image to tensor format"""
    image = image.convert('RGB').resize((WIDTH, HEIGHT))
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32)
    image = rescale(image, (0, 255), (-1, 1))
    image = image.permute(2, 0, 1)
    return image

def collate_fn(examples):
    """Custom collate function for dataloader"""
    images = []
    captions = []
    
    for example in examples:
        try:
            image = preprocess_image(example['image'])
            images.append(image)
            captions.append(example['text'])
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    if len(images) == 0:
        return None
    
    images = torch.stack(images)
    return {'images': images, 'captions': captions}

def get_time_embedding(timestep, device):
    """Get time embedding for a single timestep"""
    freqs = torch.pow(
        10000, 
        -torch.arange(start=0, end=160, dtype=torch.float32, device=device) / 160
    )
    x = torch.tensor([timestep], dtype=torch.float32, device=device)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    print(f"Using device: {DEVICE}")
    print(f"LoRA Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}")
    
    # Clear memory before starting
    clear_memory()
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train[:100]")
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Load tokenizer
    print("\n=== Loading Tokenizer ===")
    tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)
    
    # Load models
    print("\n=== Loading Models ===")
    models = preload_models_from_standard_weights(MODEL_PATH, DEVICE)
    
    # Get models
    encoder = models['encoder']
    clip = models['clip']
    diffusion = models['diffusion']
    
    # Delete decoder to free memory
    if 'decoder' in models:
        del models['decoder']
    clear_memory()
    
    # Freeze VAE and CLIP
    encoder.eval()
    clip.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    for param in clip.parameters():
        param.requires_grad = False
    
    print(f"Original diffusion parameters: {count_parameters(diffusion):,}")
    
    # Inject LoRA into diffusion model
    print("\n=== Injecting LoRA ===")
    diffusion, lora_params = inject_lora_to_diffusion_unet(
        diffusion, 
        rank=LORA_RANK, 
        alpha=LORA_ALPHA
    )
    
    # Freeze all original parameters
    for param in diffusion.parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA parameters
    for param in lora_params:
        param.requires_grad = True
    
    trainable_params = count_parameters(diffusion)
    print(f"\nTrainable parameters (LoRA only): {trainable_params:,}")
    print(f"Reduction: {100 * (1 - trainable_params / count_parameters(diffusion)):.2f}%")
    
    # Set to train mode
    diffusion.train()
    
    # Optimizer - only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=NUM_EPOCHS * len(dataloader),
        eta_min=LEARNING_RATE * 0.1
    )
    
    # Training loop
    print("\n=== Starting Training ===")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            try:
                images = batch['images'].to(DEVICE)
                captions = batch['captions']
                
                # Encode captions with CLIP
                tokens = tokenizer.batch_encode_plus(
                    captions, 
                    padding="max_length", 
                    max_length=77,
                    truncation=True
                ).input_ids
                tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    # Get text embeddings
                    context = clip(tokens)
                    
                    # Encode images to latents
                    encoder_noise = torch.randn(
                        (images.shape[0], 4, LATENTS_HEIGHT, LATENTS_WIDTH),
                        device=DEVICE
                    )
                    latents = encoder(images, encoder_noise)
                
                # Free up memory
                del images, tokens, encoder_noise
                
                # Sample random timestep
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],), 
                    device=DEVICE
                ).long()
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                
                # Get noisy latents
                sampler = DDPMSampler(torch.Generator(device=DEVICE))
                noisy_latents = sampler.add_noise(latents, timesteps)
                
                del latents
                
                # Get time embeddings
                time_embeddings = []
                for t in timesteps:
                    time_emb = get_time_embedding(t.item(), DEVICE)
                    time_embeddings.append(time_emb)
                time_embedding = torch.cat(time_embeddings, dim=0)
                
                # Forward pass
                predicted_noise = diffusion(noisy_latents, context, time_embedding)
                
                # Calculate loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Free memory
                del noisy_latents, context, time_embedding, predicted_noise, noise
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                # Save checkpoint
                if global_step % SAVE_EVERY == 0:
                    checkpoint_path = os.path.join(
                        OUTPUT_DIR, 
                        f"lora_step_{global_step}.pt"
                    )
                    save_lora_weights(diffusion, checkpoint_path)
                    print(f"\nCheckpoint saved at step {global_step}")
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    clear_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM error at batch {batch_idx}, skipping...")
                    clear_memory()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Epoch statistics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Total Batches: {num_batches}")
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(
            OUTPUT_DIR, 
            f"lora_epoch_{epoch + 1}.pt"
        )
        save_lora_weights(diffusion, epoch_checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(OUTPUT_DIR, "lora_best.pt")
            save_lora_weights(diffusion, best_path)
            print(f"  New best model saved! Loss: {best_loss:.4f}")
        
        clear_memory()
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "lora_final.pt")
    save_lora_weights(diffusion, final_path)
    print(f"\n{'='*50}")
    print("Training Completed!")
    print(f"Final model saved: {final_path}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    train()