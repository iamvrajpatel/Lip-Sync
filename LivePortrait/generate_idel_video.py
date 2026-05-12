import subprocess
import os

def generate_video(
    source_image,
    driving_video,
    output_dir="animations",
    paste_back=True
):
    """
    source_image: path to passport-style image
    driving_video: idle motion video (IMPORTANT)
    """

    if not os.path.exists(source_image):
        raise FileNotFoundError("Source image not found")

    if not os.path.exists(driving_video):
        raise FileNotFoundError("Driving video not found")

    cmd = [
        "python", "inference.py",
        "-s", source_image,
        "-d", driving_video
    ]

    if not paste_back:
        cmd.append("--no_flag_pasteback")

    print("Running LivePortrait...")
    subprocess.run(cmd, check=True)

    print(f"Done. Check output in: {output_dir}/")


if __name__ == "__main__":
    generate_video(
        source_image=r"/home/vrajpatel/Downloads/WhatsApp Image 2026-04-28 at 6.38.13 PM.jpeg",
        driving_video=r"/home/vrajpatel/personal/lip-sync/Lip-Sync/LivePortrait/assets/ideal/driving/ideal.mp4"
    )