# LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync

LatentSync is a state-of-the-art framework for generating lip-synced videos using audio-conditioned latent diffusion models. This repository provides all necessary scripts, models, and utilities for training, inference, and evaluation of lip-sync video generation.

---

## Table of Contents

- [LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync](#latentsync-audio-conditioned-latent-diffusion-models-for-lip-sync)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Quick Setup](#quick-setup)
    - [Manual Setup](#manual-setup)
  - [Usage](#usage)
    - [Inference](#inference)
    - [API Usage](#api-usage)
    - [Evaluation](#evaluation)
  - [Notes](#notes)
  - [Citation](#citation)
  - [License](#license)

---

## Features

- Audio-conditioned video generation using latent diffusion models.
- End-to-end training and inference pipelines.
- Tools for preprocessing, evaluation (FVD, SyncNet), and dataset management.
- Gradio web UI for easy demo.
- Modular and extensible codebase.

---

## Project Structure

```
lip-sync/
├── create_env.py                # Automated environment and model setup script
├── requirements.txt             # Top-level requirements (may be empty)
├── inference.py                 # Inference entry point
├── gradio_app.py                # Gradio web UI for demo
├── bg_change.py                 # Background change utility (FastAPI)
├── LatentSync/                  # Main package directory
│   ├── requirements.txt         # Project requirements
│   ├── README.md                # (This file)
│   ├── latentsync/              # Core model, pipeline, and utility code
│   ├── eval/                    # Evaluation scripts (FVD, SyncNet, etc.)
│   ├── preprocess/              # Data preprocessing scripts
│   ├── data/                    # Dataset classes
│   ├── scripts/                 # Training and inference scripts
│   ├── predict.py               # Prediction script
│   └── ...                      # Other utilities and modules
├── checkpoints/                 # Model checkpoints (downloaded by setup)
├── outputs/                     # Output directory for results
├── temp/                        # Temporary files
└── ...
```

---

## Installation

### Requirements

- **Python 3.11.9** (recommended)
- pip (latest)
- Git

### Quick Setup

The easiest way to set up the environment and download all required models is to use the provided `create_env.py` script.



**Note:** If you encounter permission issues, try running with `python -m pip ...` or as administrator.

### Manual Setup

If you prefer to install dependencies manually:

1. **Create and activate a Python 3.11.9 virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/iamvrajpatel/Lip-Sync.git
   ```

3. **Ensure you are using Python 3.11.9:**
   ```bash
   python --version
   # If not 3.11.9, create a new virtual environment with Python 3.11.9
   ```

4. **Run the environment setup script:**
   ```bash
   python create_env.py
   ```
   This script will:
   - Install all required Python packages (from `LatentSync/requirements.txt`).
   - Download all necessary model checkpoints and auxiliary files.
   - Set up directory structure.

5. **Install requirements:**
   ```bash
   pip install -r LatentSync/requirements.txt
   ```

6. **Download model checkpoints:**
   - Download the required model files as listed in `create_env.py` and place them in the appropriate directories (`checkpoints/`, `/root/.cache/torch/hub/checkpoints/`, etc.).

---

## Usage

### Inference

To generate a lip-synced video:

```bash
python inference.py --video_path <input_video.mp4> --audio_path <input_audio.wav> --video_out_path <output_video.mp4> --inference_ckpt_path <checkpoint_path>
```

Or use the Gradio web UI:

```bash
python gradio_app.py   # In /LatentSync Folder
```
This will launch a browser interface for uploading video and audio files.

### API Usage

You can also run the FastAPI server using `main.py` and interact with it via HTTP requests.

**Start the API server:**
```bash
python main.py
```
This will start a FastAPI server at `http://127.0.0.1:8000`.

**Example: Generate a video using the API with `curl`:**
```bash
curl --location 'http://127.0.0.1:8000/generate-video' \
--form 'video=@"<video-path>"' \
--form 'audio=@"<audio-path>"' \
--form 'seed="1247"' \
--form 'num_steps="40"' \
--form 'guidance_scale="1.0"' \
--form 'video_scale="0.8"' \
--form 'output_fps="30"'
```
Replace the file paths with your own video and audio files.

### Evaluation

Evaluation scripts are provided in the `LatentSync/eval/` directory, including FVD and SyncNet-based metrics. See the respective scripts for usage instructions.

---

## Notes

- All major scripts and modules are inside the `LatentSync/` directory.
- For training, preprocessing, and advanced usage, refer to scripts in `LatentSync/scripts/` and `LatentSync/preprocess/`.
- Checkpoints and large model files are automatically downloaded by `create_env.py`.

---

## Citation

If you use this codebase, please cite the corresponding paper (see [arXiv link](https://arxiv.org/pdf/2412.09262)).

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
