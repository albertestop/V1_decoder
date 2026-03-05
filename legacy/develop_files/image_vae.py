from pathlib import Path
from diffusers import models
from PIL import Image
import torch, numpy as np
from torchvision import transforms



def enc_dec_img(current_path):
    """
    Takes an RGB image and compresses it and decompresses it.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = models.AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        token=True
    ).to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),              # HWC → CHW [0‒1]
        transforms.Resize(1024,             # SD‑3 Medium is 1024 px natively;
                        interpolation=Image.BICUBIC,
                        antialias=True),
        transforms.CenterCrop(1024),
        transforms.Normalize([0.5], [0.5])  # [0‒1] → [‑1‒+1]
    ])


    img = Image.open(current_path / Path("img_sample.png")).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)          # (1,3,1024,1024)

    with torch.no_grad():
        posterior = vae.encode(x)                        # includes μ, σ
        latents   = posterior.latent_dist.sample()       # (1,4,128,128)
        latents  *= vae.config.scaling_factor            # scale = 0.18215

    torch.save(latents.half().cpu(), current_path / Path("my_picture.latents.pt"))

    # Decode latent back to RGB

    latents = torch.load(current_path / Path("my_picture.latents.pt"), map_location=device)
    latents = latents.float() / vae.config.scaling_factor

    with torch.no_grad():
        recon = vae.decode(latents).sample               # (1,3,1024,1024)
        recon = recon.clamp(-1,1).add(1).div(2)          # → [0,1]

    # Save / show
    recon_img = transforms.ToPILImage()(recon[0].cpu())
    recon_img.save(current_path / Path("my_picture_reconstructed_sample.png"))


if __name__ == "__main__":
    current_path = Path(__file__).parent
    enc_dec_img(current_path)
