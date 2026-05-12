#install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# shared app dependencies for TTS and voice cloning
pip install parler-tts TTS soundfile


# location :=> LivePortrait/

# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"


# Additinal installations
pip install pykalman==0.9.7 onnx tyro imageio[ffmpeg]
