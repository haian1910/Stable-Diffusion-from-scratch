import argparse
import os
from pathlib import Path
from PIL import Image
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer

import model_loader
from ddpm import DDPMSampler

# Training script that: loads images + captions (each image may have a .txt with the same basename),
# loads pretrained models via model_loader.preload_models_from_standard_weights, freezes VAE and CLIP,
# trains only the Diffusion model to predict the noise (MSE) and saves a checkpoint.

# Defaults
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def download_and_extract(url: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    content = r.content
    # Try zip
    if url.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(dest)
        return
    # Try tar
    if url.endswith(".tar.gz") or url.endswith(".tgz") or tarfile.is_tarfile(io.BytesIO(content)):
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tf:
            tf.extractall(dest)
        return
    # Not an archive: save as-is
    # Save single file
    out_path = dest / Path(url).name
    with open(out_path, "wb") as f:
        f.write(content)


class NarutoDataset(Dataset):
    def __init__(self, split="train", image_size=(WIDTH, HEIGHT)):
        self.dataset = load_dataset("lambdalabs/naruto-blip-captions", split=split)
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # to [-1, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # HF dataset returns PIL Image for 'image' column
        img = item['image']
        img = self.transform(img)
        return img, item['text']


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return imgs, captions


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    print("Loading Naruto BLIP captions dataset...")
    dataset = NarutoDataset(split=args.split, image_size=(WIDTH, HEIGHT))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    print("Loading models from checkpoint...")
    models = model_loader.preload_models_from_standard_weights(args.ckpt_path, device)

    clip = models['clip']
    encoder = models['encoder']
    diffusion = models['diffusion']

    # Freeze VAE (encoder + decoder); here we only need encoder to produce latents for training
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Freeze CLIP
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    # Move diffusion to train mode and to device (already in device from loader, but ensure)
    diffusion.train()
    diffusion.to(device)

    # Setup tokenizer
    tokenizer = CLIPTokenizer(args.tokenizer_vocab, merges_file=args.tokenizer_merges)

    # Sampler helper for alphas
    sampler = DDPMSampler(torch.Generator(), num_training_steps=args.num_train_timesteps)
    alphas_cumprod = sampler.alphas_cumprod.to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, captions in pbar:
            imgs = imgs.to(device)
            b = imgs.shape[0]

            # Encode images to latents using encoder (encoder needs a noise input per forward API)
            # The encoder expects noise in latent space shape (B, 4, H/8, W/8)
            encoder_noise = torch.randn((b, 4, LATENTS_HEIGHT, LATENTS_WIDTH), device=device)
            with torch.no_grad():
                latents = encoder(imgs, encoder_noise)  # already scaled inside encoder

            # Sample random timesteps for each example in the batch
            timesteps = torch.randint(0, sampler.num_train_timesteps, (b,), device=device).long()

            # Prepare noise and noised latents (vectorized)
            alphas = alphas_cumprod[timesteps].view(b, *([1] * (latents.dim() - 1)))
            sqrt_alpha = torch.sqrt(alphas)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas)
            noise = torch.randn_like(latents)
            noisy_latents = sqrt_alpha * latents + sqrt_one_minus_alpha * noise

            # Prepare context (CLIP embeddings)
            # Tokenize captions
            batch_tokens = tokenizer(captions, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
            input_ids = batch_tokens['input_ids'].to(device)
            with torch.no_grad():
                context = clip(input_ids)

            # Build time embeddings: diffusion.TimeEmbedding expects shape (1, 320) as in pipeline; the same is used for all batch
            # We will compute time embedding per unique timestep to save computation
            unique_ts, inverse_indices = torch.unique(timesteps, return_inverse=True)
            time_emb_cache = {}
            for i, t in enumerate(unique_ts.tolist()):
                te = get_time_embedding(t).to(device)
                time_emb_cache[t] = te

            # Map each batch element to its time embedding
            # We'll build a list of time embeddings per-sample and then stack? Many implementations broadcast a (1,320) embedding.
            # Our diffusion implementation expects a single time embedding (likely broadcastable). We'll pick per-batch embeddings by indexing cache.

            # Run model: diffusion expects model_input shape (B, 4, H/8, W/8); for classifier-free guidance we'd double batch, but training here is unconditional.
            optimizer.zero_grad()

            # For vectorized call we need to pass a time embedding - diffusion code likely expects a (1, 320) or (B, 320) tensor.
            # We'll pass per-sample embeddings by stacking along batch dimension.
            time_embs = torch.cat([time_emb_cache[int(t.item())] for t in timesteps], dim=0)  # (B, 320)

            # Forward pass
            pred_noise = diffusion(noisy_latents, context, time_embs)

            loss = mse(pred_noise, noise)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'step': global_step})

            # Save checkpoint periodically
            if global_step % args.save_every == 0:
                out = {
                    'diffusion': diffusion.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_step': global_step,
                }
                out_path = Path(args.out_dir) / f"checkpoint-step-{global_step}.pt"
                torch.save(out, out_path)
                print(f"Saved checkpoint to {out_path}")

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final save
    out = {
        'diffusion': diffusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
    }
    out_path = Path(args.out_dir) / "checkpoint-final.pt"
    torch.save(out, out_path)
    print(f"Training finished. Saved final checkpoint to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use (train/validation/test)')
    parser.add_argument('--ckpt_path', type=str, default='/content/Stable-Diffusion-from-scratch/data/v1-5-pruned-emaonly.ckpt', help='Path to standard weights checkpoint to load (same used by model_loader).')
    parser.add_argument('--tokenizer_vocab', type=str, default='/content/Stable-Diffusion-from-scratch/data/vocab.json', help='Path to tokenizer vocab.json')
    parser.add_argument('--tokenizer_merges', type=str, default='/content/Stable-Diffusion-from-scratch/data/merges.txt', help='Path to tokenizer merges.txt')
    parser.add_argument('--out_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=200, help='Steps between saves')
    parser.add_argument('--max_steps', type=int, default=0, help='Stop after this many steps (0 means no limit)')
    parser.add_argument('--num_train_timesteps', type=int, default=1000)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)