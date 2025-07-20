# @title
import torch

if torch.cuda.is_available():
    import gc
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
        
        
# import ipywidgets as widgets
import torch
import torchaudio
import subprocess
from datetime import datetime
import os
import ffmpeg
loop_vid_from_endframe = True # @param {"type":"boolean"}

def convert_video_fps(input_path, target_fps):
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        print(f"Error: The video file {input_path} is missing or empty.")
        return None

    output_path = f"converted_{target_fps}fps.mp4"

    audio_check_cmd = [
        "ffprobe", "-i", input_path, "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    audio_present = subprocess.run(audio_check_cmd, capture_output=True, text=True).stdout.strip() != ""

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:v", f"fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]

    if audio_present:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    subprocess.run(cmd, check=True)
    print(f"Converted video saved as {output_path}")
    return output_path


def trim_video(video_path, target_duration):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        print(f"Error: The video file {video_path} is missing or empty.")
        return video_path

    has_audio = False
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='a:0', show_entries='stream=codec_type')
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    except ffmpeg.Error as e:
        print(f"Error while probing video: {e}")
        return video_path

    trimmed_video_path = "trimmed_video.mp4"
    try:
        if has_audio:
            ffmpeg.input(video_path, ss=0, to=target_duration).output(trimmed_video_path, codec="libx264", audio_codec="aac").run()
        else:
            ffmpeg.input(video_path, ss=0, to=target_duration).output(trimmed_video_path, codec="libx264").run()
        print("Video trimmed")
    except ffmpeg.Error as e:
        print(f"Error during video trimming: {e}")
        return video_path

    return trimmed_video_path


def has_audio(video_path):
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='a', show_entries='stream=index')
        return len(probe['streams']) > 0
    except ffmpeg.Error:
        return False

def extend_video(video_path, target_duration):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        print(f"Error: The video file {video_path} is missing or empty.")
        return video_path

    audio_exists = has_audio(video_path)

    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        original_duration = float(probe['format']['duration'])
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "No error message available"
        print(f"Error: Unable to fetch video duration: {error_message}")
        # print(f"Error: Unable to fetch video duration: {e.stderr.decode()}")
        return video_path

    if original_duration <= 0:
        print("Error: Invalid video duration!")
        return video_path

    print("Extending video...")

    clips = [video_path]
    total_duration = original_duration
    extensions = 0

    while total_duration < target_duration:
        extensions += 1
        if loop_vid_from_endframe:
            reversed_clip = reverse_video(clips[-1], audio_exists)
            clips.append(reversed_clip)
        else:
            clips.append(clips[-1])
            # new_clip = f"copy_{extensions}_{os.path.basename(clips[-1])}"
            # shutil.copy(clips[-1], new_clip)
            # clips.append(new_clip)
        total_duration += original_duration

    print(f"The video was extended {extensions} time(s)")

    extended_video_path = "extended_video.mp4"

    try:
        inputs = [ffmpeg.input(clip) for clip in clips]

        if audio_exists:
            concat = ffmpeg.concat(*inputs, v=1, a=1).output(extended_video_path, codec="libx264", audio_codec="aac", format="mp4", vcodec="libx264", acodec="aac")
        else:
            concat = ffmpeg.concat(*inputs, v=1, a=0).output(extended_video_path, codec="libx264", format="mp4", vcodec="libx264")

        concat.run(overwrite_output=True)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "No error message available"
        print(f"Error during video concatenation: {error_message}")
        return video_path

    for clip in clips[1:]:
        if os.path.exists(clip):
            os.remove(clip)

    return extended_video_path


def reverse_video(video_path, audio_exists):

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    reversed_video_path = f"reversed_{os.path.basename(video_path)}"
    # reversed_video_path = os.path.join(
    #     os.path.dirname(video_path),
    #     f"r_{timestamp}_{os.path.basename(video_path)}"
    # )
    try:
        if audio_exists:
            ffmpeg.input(video_path).output(reversed_video_path, vf='reverse', af='areverse').run(overwrite_output=True)
        else:
            ffmpeg.input(video_path).output(reversed_video_path, vf='reverse').run(overwrite_output=True)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "No error message available"
        print(f"Error during video reversal: {error_message}")
        return video_path

    return reversed_video_path


def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Error: Unable to fetch video duration for {video_path}: {e}")
        return 0


def pad_audio_to_multiple_of_16_for_video(audio_path, target_fps=25):
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sample_rate
    num_frames = int(audio_duration * target_fps)
    remainder = num_frames % 16

    if remainder > 0:
        pad_frames = 16 - remainder
        pad_samples = int((pad_frames / target_fps) * sample_rate)
        pad_waveform = torch.zeros((waveform.shape[0], pad_samples))
        waveform = torch.cat((waveform, pad_waveform), dim=1)
        padded_audio_path = "padded_audio.wav"
        torchaudio.save(padded_audio_path, waveform, sample_rate)
    else:
        padded_audio_path = audio_path

    return padded_audio_path, int((waveform.shape[1] / sample_rate) * target_fps), waveform.shape[1] / sample_rate








# Rewriting some functions

def trim_video(video_path, target_duration):
    """Trim video to specified duration with robust error handling"""
    # Validate input file
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return video_path
    if os.path.getsize(video_path) == 0:
        print(f"Error: Video file is empty at {video_path}")
        return video_path
    if target_duration <= 0:
        print(f"Error: Invalid target duration {target_duration}")
        return video_path

    # Get original duration for validation
    try:
        probe = ffmpeg.probe(video_path, v='error', show_entries='format=duration')
        original_duration = float(probe['format']['duration'])
        if original_duration <= 0:
            print("Error: Could not determine valid video duration")
            return video_path

        print(f"Original duration: {original_duration:.2f}s, Target duration: {target_duration:.2f}s")

        if original_duration <= target_duration:
            print("Video is already shorter than target duration, no trimming needed")
            return video_path
    except Exception as e:
        print(f"Error probing video duration: {str(e)}")
        return video_path

    # Check for audio stream more robustly
    has_audio = False
    try:
        audio_probe = ffmpeg.probe(
            video_path,
            v='error',
            select_streams='a',
            show_entries='stream=codec_type,codec_name'
        )
        has_audio = any(stream['codec_type'] == 'audio' for stream in audio_probe.get('streams', []))
        if has_audio:
            audio_codec = audio_probe['streams'][0]['codec_name']
            print(f"Detected audio stream with codec: {audio_codec}")
    except Exception as e:
        print(f"Warning: Could not determine audio status: {str(e)}")

    # Prepare output path with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trimmed_video_path = f"trimmed_{timestamp}.mp4"

    # Build ffmpeg command
    try:
        input_stream = ffmpeg.input(video_path, ss=0, to=target_duration)

        output_args = {
            'c:v': 'libx264',
            'preset': 'fast',
            'crf': '18',
            'pix_fmt': 'yuv420p',
            'movflags': '+faststart'  # For web optimization
        }

        if has_audio:
            output_args['c:a'] = 'aac'
            output_args['b:a'] = '192k'
            output_args['ar'] = '44100'
            output_args['ac'] = '2'  # Stereo audio

        # Use subprocess for better error handling
        cmd = (
            input_stream
            .output(trimmed_video_path, **output_args)
            .compile()
        )

        # print(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("Video trimmed successfully")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg trimming failed with error:\n{e.stderr}")
        # Clean up potentially partial output file
        if os.path.exists(trimmed_video_path):
            try:
                os.remove(trimmed_video_path)
            except Exception as clean_err:
                print(f"Warning: Could not clean up failed output: {str(clean_err)}")
        return video_path
    except Exception as e:
        print(f"Unexpected error during trimming: {str(e)}")
        return video_path

    # Verify output
    if not os.path.exists(trimmed_video_path):
        print("Error: Trimmed video file was not created")
        return video_path

    if os.path.getsize(trimmed_video_path) == 0:
        print("Error: Trimmed video file is empty")
        os.remove(trimmed_video_path)
        return video_path

    try:
        output_duration = float(ffmpeg.probe(trimmed_video_path)['format']['duration'])
        duration_diff = abs(output_duration - target_duration)
        if duration_diff > 0.5:  # Allow 0.5s tolerance
            print(f"Warning: Trimmed duration is {output_duration:.2f}s (target: {target_duration:.2f}s)")
    except Exception as e:
        print(f"Warning: Could not verify output duration: {str(e)}")

    return trimmed_video_path

def extend_video(video_path, target_duration):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        print(f"Error: The video file {video_path} is missing or empty.")
        return video_path

    # Check audio existence more robustly
    audio_exists = has_audio(video_path)
    print(f"Audio exists in source: {audio_exists}")

    # Get original duration with verification
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        original_duration = float(probe['format']['duration'])
        print(f"Original duration: {original_duration:.2f}s, Target duration: {target_duration:.2f}s")

        if original_duration <= 0:
            raise ValueError("Invalid video duration detected")
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return video_path

    # Calculate needed extensions
    if original_duration >= target_duration:
        print("Video already meets target duration")
        return video_path

    clips = [video_path]
    total_duration = original_duration
    extensions = 0

    # Create extended clips
    while total_duration < target_duration:
        extensions += 1
        try:
            if loop_vid_from_endframe:
                reversed_clip = reverse_video(clips[-1], audio_exists)
                if not os.path.exists(reversed_clip) or os.path.getsize(reversed_clip) == 0:
                    raise Exception("Reversed clip creation failed")
                clips.append(reversed_clip)
                # print(f"Created reversed clip: {reversed_clip}")
            else:
                clips.append(clips[-1])

            total_duration += original_duration
            # print(f"Extended to {total_duration:.2f}s (iteration {extensions})")
        except Exception as e:
            print(f"Failed during clip extension: {str(e)}")
            break

    # Verify we actually extended the video
    if len(clips) <= 1:
        print("No extension performed, returning original")
        return video_path

    # Check all clips before concatenation
    print("\nClip properties before concatenation:")
    for i, clip in enumerate(clips):
        try:
            probe = ffmpeg.probe(clip)
            # print(f"Clip {i+1}: {os.path.basename(clip)}")
            # print(f"  Size: {os.path.getsize(clip)/1024/1024:.2f}MB")
            for stream in probe['streams']:
                if stream['codec_type'] == 'video':
                    print(f"  Video: {stream['codec_name']} {stream['width']}x{stream['height']}")
                elif stream['codec_type'] == 'audio':
                    print(f"  Audio: {stream['codec_name']}")
        except Exception as e:
            print(f"Error checking clip {clip}: {str(e)}")
            return video_path

    # Concatenation using demuxer method (most reliable)
    extended_video_path = "extended_video.mp4"
    concat_list_path = "concat_list.txt"

    try:
        # Create concat list file
        with open(concat_list_path, 'w') as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy'  # Stream copy (no re-encoding)
        ]

        # For some formats, we need to force MP4 output
        if not extended_video_path.endswith('.mp4'):
            cmd.extend(['-f', 'mp4'])

        cmd.append(extended_video_path)

        # Run command with error capture
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Concatenation successful!")

    except subprocess.CalledProcessError as e:
        print(f"Concatenation failed with error:\n{e.stderr}")
        return video_path
    except Exception as e:
        print(f"Unexpected error during concatenation: {str(e)}")
        return video_path
    finally:
        # Cleanup temporary files
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)

        # Remove intermediate reversed clips
        for clip in clips[1:]:
            if os.path.exists(clip):
                try:
                    os.remove(clip)
                except Exception as e:
                    print(f"Warning: Could not remove {clip}: {str(e)}")

    # Verify output
    if not os.path.exists(extended_video_path) or os.path.getsize(extended_video_path) == 0:
        print("Error: Final extended video not created properly")
        return video_path

    final_duration = get_video_duration(extended_video_path)
    print(f"Final extended duration: {final_duration:.2f}s")

    return extended_video_path


def reverse_video(video_path, audio_exists):
    """Create a reversed version of the video"""
    reversed_path = f"reversed_{os.path.basename(video_path)}"
    try:
        if audio_exists:
            (
                ffmpeg.input(video_path)
                .output(reversed_path, vf='reverse', af='areverse')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        else:
            (
                ffmpeg.input(video_path)
                .output(reversed_path, vf='reverse')
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
        return reversed_path
    except ffmpeg.Error as e:
        print(f"Reverse failed: {e.stderr.decode()}")
        raise

def has_audio(video_path):
    """Check if video contains audio stream"""
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='a')
        return len(probe['streams']) > 0
    except ffmpeg.Error:
        return False

def get_video_duration(video_path):
    """Get duration in seconds"""
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='format=duration')
        return float(probe['format']['duration'])
    except Exception as e:
        print(f"Duration check failed: {str(e)}")
        return 0
# End of new functions



# app.py
import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch

from inference import perform_inference

# --- import or re-implement these helper functions in your module ---
# convert_video_fps(input_path, target_fps) → str
# pad_audio_to_multiple_of_16(audio_path, target_fps) → (padded_audio_path, n_frames, duration_sec)
# get_video_duration(video_path) → duration_sec
# extend_video(video_path, target_duration_sec) → str
# trim_video(video_path, target_duration_sec) → str
# perform_inference(video_path, audio_path, seed, num_steps, guidance_scale, output_path)
# -------------------------------------------------------------------

app = FastAPI()

# Allow CORS if you’re calling from a web front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or lock down to your domains
    allow_methods=["POST"],
    allow_headers=["*"],
)

def save_upload(tmp_dir: str, upload: UploadFile) -> str:
    """Save UploadFile to disk and return path."""
    ext = os.path.splitext(upload.filename)[1]
    out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}{ext}")
    with open(out_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out_path

@app.get("/")
async def root():
    return {"status": "Lip-Sync Running!!"}

@app.post("/generate-from-video")
async def lip_sync(
    video: UploadFile = File(..., description=".mp4 file"),
    audio: UploadFile = File(..., description=".wav/.mp3/.aac/.flac"),
    seed: int = Form(1247),
    num_steps: int = Form(20, ge=1, le=100),
    guidance_scale: float = Form(1.0, ge=0.1, le=10.0),
    video_scale: float = Form(0.5, ge=0.1, le=1.0),
    output_fps: int = Form(25, ge=6, le=60),
):
    # create isolated work dir
    job_id = str(uuid.uuid4())
    tmp_dir = os.path.join("tmp", job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # 1) Save uploads
        video_path = save_upload(tmp_dir, video)
        audio_path = save_upload(tmp_dir, audio)

        # 2) Determine width/height
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Invalid video file")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception as e:
            raise RuntimeError(f"Error while determining height-weight : {e}")

        # 3) Normalize video FPS to 25 for processing
        try:
            proc_video = convert_video_fps(video_path, 25)
        except Exception as e:
            raise RuntimeError(f"Error while normalizing video FPS : {e}")

        # 4) Pad audio & get durations
        proc_audio, num_frames, audio_dur = pad_audio_to_multiple_of_16_for_video(audio_path, target_fps=25)
        video_dur = get_video_duration(proc_video)

        # 5) Sync lengths
        if audio_dur > video_dur:
            proc_video = extend_video(proc_video, audio_dur)
            video_dur = get_video_duration(proc_video)
            if video_dur > audio_dur:
                proc_video = trim_video(proc_video, audio_dur)
        elif video_dur > audio_dur:
            proc_video = trim_video(proc_video, audio_dur)

        # 6) Inference
        output_path = os.path.join(tmp_dir, "output_video.mp4")
        perform_inference(
            proc_video,
            proc_audio,
            seed,
            num_steps,
            guidance_scale,
            output_path,
        )

        # 7) Final FPS conversion
        final_path = convert_video_fps(output_path, output_fps)

        # 8) Return
        return FileResponse(final_path, media_type="video/mp4", filename="output_video.mp4")

    finally:
        # cleanup GPU caches and tmp files
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ==========================================================
# **************** VIDEO FROM IMAGE ***********************
# ==========================================================


def convert_video_fps(input_path, target_fps):
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        print(f"Error: The video file {input_path} is missing or empty.")
        return None

    output_path = f"converted_{target_fps}fps.mp4"

    audio_check_cmd = [
        "ffprobe", "-i", input_path, "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    audio_present = subprocess.run(audio_check_cmd, capture_output=True, text=True).stdout.strip() != ""

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:v", f"fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]

    if audio_present:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    subprocess.run(cmd, check=True)
    print(f"Converted video saved as {output_path}")
    return output_path

# def add_silent_frames(audio_path, target_fps=25):

#     waveform, sample_rate = torchaudio.load(audio_path)
#     silent_duration = 25 / target_fps  # Two frames at target FPS
#     silent_samples = int(silent_duration * sample_rate)
#     silent_waveform = torch.zeros((waveform.shape[0], silent_samples))

#     # Concatenate silence at the beginning for mouth correction
#     new_waveform = torch.cat((silent_waveform, waveform), dim=1)
#     new_audio_path = "audio_with_silence.wav"
#     torchaudio.save(new_audio_path, new_waveform, sample_rate)

#     return new_audio_path


def pad_audio_to_multiple_of_16_for_audio(audio_path, target_fps=25):

    # audio_path = add_silent_frames(audio_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sample_rate  # Duration in seconds

    num_frames = int(audio_duration * target_fps)

    # Pad audio to ensure frame count is a multiple of 16
    remainder = num_frames % 16
    if remainder > 0:
        pad_frames = 16 - remainder
        pad_samples = int((pad_frames / target_fps) * sample_rate)
        pad_waveform = torch.zeros((waveform.shape[0], pad_samples))  # Silence padding
        waveform = torch.cat((waveform, pad_waveform), dim=1)

        # Save the padded audio
        padded_audio_path = "padded_audio.wav"
        torchaudio.save(padded_audio_path, waveform, sample_rate)
    else:
        padded_audio_path = audio_path  # No padding needed

    padded_duration = waveform.shape[1] / sample_rate
    padded_num_frames = int(padded_duration * target_fps)

    return padded_audio_path, padded_num_frames


def create_video_from_image(image_path, output_video_path, num_frames, fps=25):
    """Convert an image into a video of specified length (num_frames at 25 FPS)."""
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read the image.")
        return None

    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        video_writer.write(img)

    video_writer.release()
    print(f"Created video {output_video_path} with {num_frames} frames ({num_frames / fps:.2f} seconds).")
    return output_video_path

@app.post("/generate-from-image", response_class=FileResponse)
async def generate_lipsync(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    seed: int = Form(1247),
    steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    output_fps: int = Form(25),
):
    # create a temp working dir
    workdir = os.path.join("/tmp", uuid.uuid4().hex)
    os.makedirs(workdir, exist_ok=True)

    try:
        # Save uploads
        image_path = save_upload(workdir, image)
        audio_path = save_upload(workdir, audio)

        # Pad audio & get frame count
        padded_audio, num_frames = pad_audio_to_multiple_of_16_for_audio(audio_path, target_fps=output_fps)

        # Build a static video from the image
        raw_video = os.path.join(workdir, "input_video.mp4")
        create_video_from_image(image_path, raw_video, num_frames, fps=output_fps)

        # Run your inference
        generated = os.path.join(workdir, "gen_video.mp4")
        perform_inference(
            raw_video,
            padded_audio,
            seed,
            steps,
            guidance_scale,
            generated
        )

        # Re-encode at requested FPS
        final_video = convert_video_fps(generated, output_fps)
        if final_video is None or not os.path.exists(final_video):
            raise HTTPException(status_code=500, detail="Failed to convert output video")

        # Return the file
        return FileResponse(
            final_video,
            media_type="video/mp4",
            filename="lipsync_output.mp4"
        )

    finally:
        # cleanup
        shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")