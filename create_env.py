# @title
import os
import subprocess

# !pip install diffusers mediapipe transformers huggingface-hub omegaconf einops opencv-python face-alignment decord ffmpeg-python safetensors soundfile

requirements = ["pip", "install", "diffusers", "mediapipe", "transformers",
    "huggingface-hub", "omegaconf", "einops", "opencv-python", "face-alignment", "decord", 
    "ffmpeg-python", "safetensors", "soundfile"
]

subprocess.run(requirements)

print("Installed all the required Libraries!!")

if not os.path.exists("LatentSync"):
    print("Git Cloning LatentSync..!!")
    subprocess.run(["git", "clone", "https://github.com/Isi-dev/LatentSync"], shell=True)
    # !git clone https://github.com/Isi-dev/LatentSync

subprocess.run(["cd", "LatentSync"], shell=True)

# %cd LatentSync

os.makedirs("/root/.cache/torch/hub/checkpoints", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

model_urls = {
    "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth":
        "https://huggingface.co/Isi99999/LatentSync/resolve/main/auxiliary/s3fd-619a316812.pth",
    "/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip":
        "https://huggingface.co/Isi99999/LatentSync/resolve/main/auxiliary/2DFAN4-cd938726ad.zip",
    "checkpoints/latentsync_unet.pt":
        "https://huggingface.co/Isi99999/LatentSync/resolve/main/latentsync_unet.pt",
    "checkpoints/tiny.pt":
        "https://huggingface.co/Isi99999/LatentSync/resolve/main/whisper/tiny.pt",
    "checkpoints/diffusion_pytorch_model.safetensors":
        "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors",
    "checkpoints/config.json":
        "https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json",
}

import platform

os_type = platform.system()

for file_path, url in model_urls.items():
    if not os.path.exists(file_path):
        try:
            print(f"Downloading {file_path} ...")
            if os_type == "Linux":
                subprocess.run(["wget", url, "-O", file_path])   # Linux
            elif os_type == "Windows":
                subprocess.run(["curl", "-L" ,url, "-o", file_path], shell=True)   # Windows
        except Exception as e:
            raise RuntimeError(f"Error Downloading models: {e}")
    else:
        print(f"File {file_path} already exists. Skipping download.")


latent_requirements = ["pip", "install", "-r" , "LatentSync/requirements.txt"]
print("Installing Latent Requirements..!!")

subprocess.run(latent_requirements, shell=True)

print("Setup complete.")
