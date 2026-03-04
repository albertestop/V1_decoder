from pathlib import Path
from diffusers import models
from PIL import Image
import torch, numpy as np
from torchvision import transforms
import torch.nn.functional as F



def enc_dec_npy(current_path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes an 1-channel np array and compresses it and decompresses it.
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


    orig_npy = np.load(current_path / Path("img_downs.npy"))[:, :, 1]
    orig_npy_sq = np.zeros((64, 64))
    orig_npy_sq[14:50, :] = orig_npy
    orig_img = Image.fromarray(orig_npy_sq).convert("RGB")
    orig_torch = preprocess(orig_img).unsqueeze(0).to(device)          # (1,3,1024,1024)

    with torch.no_grad():
        posterior = vae.encode(orig_torch)                        # includes μ, σ
        latents   = posterior.latent_dist.sample()       # (1,4,128,128)
        latents  *= vae.config.scaling_factor            # scale = 0.18215

    torch.save(latents.half().cpu(), current_path / Path("my_picture.latents.pt"))

    # Decode latent back to RGB

    latents = torch.load(current_path / Path("my_picture.latents.pt"), map_location=device)
    latents = latents.float() / vae.config.scaling_factor

    with torch.no_grad():
        recon_torch = vae.decode(latents).sample               # (1,3,1024,1024)
        recon_torch = recon_torch.clamp(-1,1).add(1).div(2)          # → [0,1]

    # Save / show
    recon_img = transforms.ToPILImage()(recon_torch[0].cpu()).convert("L")
    recon_img.save(current_path / Path("my_picture_reconstructed_from_npy.png"))

    return orig_torch[0], recon_torch[0]


def inf_loss(orig: torch.Tensor, recons: torch.Tensor):
    orig = orig.clamp(-1,1).add(1).div(2)
    recons = recons.clamp(-1,1).add(1).div(2)

    mse = F.mse_loss(recons, orig).item()
    mae = F.l1_loss(recons, orig).item()
    print("MSE:", mse, "MAE:", mae)



if __name__ == "__main__":
    current_path = Path(__file__).parent

    orig, recons = enc_dec_npy(current_path)

    inf_loss(orig, recons)
