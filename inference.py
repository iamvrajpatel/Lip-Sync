
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from LatentSync.latentsync.models.unet import UNet3DConditionModel
from LatentSync.latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from LatentSync.latentsync.whisper.audio2feature import Audio2Feature
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed


def perform_inference(video_path, audio_path, seed=1247, num_inference_steps=20, guidance_scale=1.0, output_path="output_video.mp4"):
    config_path = "configs/unet/first_stage.yaml"
    inference_ckpt_path = "checkpoints/latentsync_unet.pt"

    config = OmegaConf.load(config_path)

    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    scheduler = DDIMScheduler.from_pretrained("configs")

    whisper_model_path = "checkpoints/tiny.pt"
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("checkpoints", torch_dtype=dtype, local_files_only=True)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        print('x_formers available!')

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    set_seed(seed)

    pipeline(
        video_path=video_path,
        audio_path=audio_path,
        video_out_path=output_path,
        video_mask_path=output_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )
    return output_path
