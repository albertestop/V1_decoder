from pathlib import Path
from diffusers import models
from PIL import Image
import torch, numpy as np
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import random 


def enc_dec_npy(current_path, img_path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes an 1-channel np array and compresses it and decompresses it.
    """

    height, width = 144, 256

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = models.AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        token=True
    ).to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),              # HWC → CHW [0‒1]
        transforms.Resize((height, width),
                        interpolation=Image.BICUBIC,
                        antialias=True),
        transforms.Normalize([0.5], [0.5])  # [0‒1] → [‑1‒+1]
    ])


    orig_img = Image.open(img_path)
    orig_npy = np.array(orig_img)
    orig_npy_sq = np.zeros((height, width))
    orig_npy_sq[:, :] = orig_npy
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
        recon_torch = recon_torch.clamp(-1,1)

    # Save / show
    recon_img = recon_torch.add(1).div(2)
    recon_img = transforms.ToPILImage()(recon_img[0].cpu()).convert("L")
    recon_img.save(current_path / Path("my_picture_reconstructed_from_orig.png"))

    # Ratio
    # print(orig_torch.shape, latents.shape)
    recons_ratio = (orig_torch.shape[2] * orig_torch.shape[3]) / (latents.shape[1] * latents.shape[2] * latents.shape[3])

    return orig_torch[0], recon_torch[0], recons_ratio


def spectrum_acc(orig, recons, cutoff=0.25, eps=1e-8):
    """
        1. 2d fourier transform and arranges values so that low freq values are in the middle 
            of the 2d space and as we get away we get to the high freq values
        2. Low freq mask takes half of the values that are closer to the center
            High freq mask takes the half of the values that are further from the center
        3. We compute the mean squared error between all the hf values and lf values
    """

    H, W = orig.shape
    cy, cx = H//2, W//2
    r = int(min(cy, cx) * cutoff)

    # FFT
    f1 = np.fft.fftshift(np.fft.fft2(orig))
    f2 = np.fft.fftshift(np.fft.fft2(recons))

    mag1, mag2 = np.abs(f1), np.abs(f2)

    # Mask
    y, x = np.ogrid[:H, :W]
    mask_low = (x-cx)**2 + (y-cy)**2 <= r**2
    mask_high = ~mask_low

    # Similarities
    diff_low = np.linalg.norm(mag1[mask_low] - mag2[mask_low])
    norm_low = np.linalg.norm(mag1[mask_low])
    S_low = 1 - diff_low / (norm_low + eps)

    diff_high = np.linalg.norm(mag1[mask_high] - mag2[mask_high])
    norm_high = np.linalg.norm(mag1[mask_high])
    S_high = 1 - diff_high / (norm_high + eps)

    return S_low, S_high


def plot_result(img1, img2, stats, i):
    img = np.concatenate([img1, img2], axis=0)
    caption = f"MSE: {stats[0]}, MAE: {stats[1]}, \nLow freq acc: {stats[2]}, High freq acc: {stats[3]}, \nSSIM: {stats[4]}, Compression ratio: {stats[5]}"

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.text(0.5, -0.1, caption,
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=9)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.savefig(f'examples/recons_stats_{i}.png')



def recons_stats(orig: torch.Tensor, recons: torch.Tensor, recons_ratio: np.float32, i: int):
    orig = orig.add(1).div(2)
    r, g, b = orig[0], orig[1], orig[2]
    orig = 0.299 * r + 0.587 * g + 0.114 * b
    recons = recons.add(1).div(2)
    r, g, b = recons[0], recons[1], recons[2]
    recons = 0.299 * r + 0.587 * g + 0.114 * b

    mse = F.mse_loss(recons, orig).item()
    mae = F.l1_loss(recons, orig).item()
    S_low, S_high = spectrum_acc(orig.cpu().detach().numpy(), recons.cpu().detach().numpy())
    SSIM, diff = ssim(
        orig.cpu().detach().numpy(), recons.cpu().detach().numpy(), full=True, 
        data_range=np.float32(recons.cpu().detach().numpy().max() - recons.cpu().detach().numpy().min()))
    # print("MSE:", mse, "MAE:", mae)
    # print("Low freq acc:", S_low, "High freq acc:", S_high)
    # print("SSIM:", SSIM)
    # print("Compression ratio: ", recons_ratio)
    stats = [mse, mae, S_low, S_high, SSIM, recons_ratio]
    # plot_result(orig.cpu().detach().numpy(), recons.cpu().detach().numpy(), stats, i)
    return stats



if __name__ == "__main__":
    path = '/data/Remote_Repository/bv_resources/all_movie_clips_bv_sets/001/'
    trial_vid_id = [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]
    images = []
    for i in range(5000):
        trial = random.choice(trial_vid_id)
        frames = [f for f in os.listdir(path + trial) if os.path.isfile(os.path.join(path + trial, f))]
        frame = random.choice(frames)
        images.append(path + trial + '/' + frame)

    current_path = Path(__file__).parent

    stats = []
    for i, img_path in enumerate(images):
        print(i)

        orig, recons, recons_ratio = enc_dec_npy(current_path, img_path)

        stats.append(recons_stats(orig, recons, recons_ratio, i))

    stats = np.array(stats, dtype=np.float32)
    stats = stats[stats[:, 3] > 0]  # why do some Low_f give -1.3e-7
    print(stats.shape)
    mean_stats = stats.mean(axis=0)
    print(mean_stats)
    np.savetxt('mean_stats.txt', mean_stats)
    
