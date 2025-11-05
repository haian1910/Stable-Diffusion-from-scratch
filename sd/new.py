import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
import os

# Import your SD modules
from sd.model_loader import preload_models_from_standard_weights
from sd.ddpm import DDPMSampler

# Configuration
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2
SAVE_EVERY = 100  # Save checkpoint every N steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# Paths
MODEL_PATH = "/content/Stable-Diffusion-from-scratch/data/v1-5-pruned-emaonly.ckpt"
VOCAB_PATH = "/content/Stable-Diffusion-from-scratch/data/vocab.json"
MERGES_PATH = "/content/Stable-Diffusion-from-scratch/data/merges.txt"
OUTPUT_DIR = "./checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rescale(x, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    return x

def preprocess_image(image):
    """Preprocess image to tensor format"""
    # Resize to 512x512
    image = image.convert('RGB').resize((WIDTH, HEIGHT))
    # Convert to numpy array
    image = np.array(image)
    # Convert to tensor and normalize to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32)
    image = rescale(image, (0, 255), (-1, 1))
    # Change from (H, W, C) to (C, H, W)
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

def train():
    print(f"Using device: {DEVICE}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train[:100]")
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)
    
    # Load models
    print("Loading models...")
    models = preload_models_from_standard_weights(MODEL_PATH, DEVICE)
    
    # Freeze VAE and CLIP (only train diffusion model)
    encoder = models['encoder']
    decoder = models['decoder']
    clip = models['clip']
    diffusion = models['diffusion']
    
    encoder.eval()
    decoder.eval()
    clip.eval()
    
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    for param in clip.parameters():
        param.requires_grad = False
    
    # Only diffusion model is trainable
    diffusion.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            if batch is None:
                continue
                
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
            
            # Sample random timestep for each image
            timesteps = torch.randint(
                0, 1000, (images.shape[0],), 
                device=DEVICE
            ).long()
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            
            # Get noisy latents using forward diffusion
            sampler = DDPMSampler(torch.Generator(device=DEVICE))
            noisy_latents = sampler.add_noise(latents, timesteps)
            
            # Get time embeddings
            time_embeddings = []
            for t in timesteps:
                freqs = torch.pow(
                    10000, 
                    -torch.arange(start=0, end=160, dtype=torch.float32, device=DEVICE) / 160
                )
                x = torch.tensor([t.item()], dtype=torch.float32, device=DEVICE)[:, None] * freqs[None]
                time_emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
                time_embeddings.append(time_emb)
            time_embedding = torch.cat(time_embeddings, dim=0)
            
            # Predict noise
            predicted_noise = diffusion(noisy_latents, context, time_embedding)
            
            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Save checkpoint
            if global_step % SAVE_EVERY == 0:
                checkpoint_path = os.path.join(
                    OUTPUT_DIR, 
                    f"diffusion_step_{global_step}.pt"
                )
                torch.save(diffusion.state_dict(), checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_checkpoint_path = os.path.join(
            OUTPUT_DIR, 
            f"diffusion_epoch_{epoch + 1}.pt"
        )
        torch.save(diffusion.state_dict(), epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "diffusion_final.pt")
    torch.save(diffusion.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved: {final_path}")

if __name__ == "__main__":
    train()