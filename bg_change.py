# main.py

import uuid
import shutil
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from rembg import remove
from tempfile import NamedTemporaryFile
from tqdm import tqdm  # ← for progress bar

app = FastAPI()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(input_path: str, bg_path: str, output_path: str):
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    # Fetch video properties
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load & resize background
    bg_img = cv2.imread(bg_path)
    if bg_img is None:
        raise RuntimeError("Could not read background image")
    bg_img = cv2.resize(bg_img, (width, height))

    try:
        # Process frames with a tqdm progress bar
        print(f"Starting processing: {frame_count} frames total")
        for _ in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # Remove background
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgba      = remove(frame_rgb)  # RGBA numpy array

                alpha = rgba[..., 3] / 255.0
                fg    = rgba[..., :3]
            except Exception as e:
                raise RuntimeError(f"Error while Removing Background: \n\n {e}")

            # Composite
            bg_rgb       = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            comp_rgb     = (alpha[..., None] * fg + (1 - alpha[..., None]) * bg_rgb).astype(np.uint8)
            comp_bgr     = cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR)

            out.write(comp_bgr)
    except Exception as e:
        raise RuntimeError(f"Error while Changing background process : \n\n {e}")

    cap.release()
    out.release()
    print("✅ Processing complete, video saved to", output_path)


@app.post("/process-video")
async def process_video_endpoint(
    video_file: UploadFile = File(..., description="Your input MP4 video"),
    bg_file:    UploadFile = File(None, description="Optional background image (jpg/png)")
):
    # Validate video
    if video_file.content_type not in ("video/mp4", "video/x-matroska"):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # Save uploaded video to temp file
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp_vid:
        shutil.copyfileobj(video_file.file, tmp_vid)
        tmp_vid_path = tmp_vid.name

    # Save or generate background
    if bg_file:
        ext = os.path.splitext(bg_file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp_bg:
            shutil.copyfileobj(bg_file.file, tmp_bg)
            tmp_bg_path = tmp_bg.name
    else:
        # Solid white default
        tmp_bg_path = os.path.join(OUTPUT_DIR, "solid_white.jpg")
        if not os.path.exists(tmp_bg_path):
            # assume 640×480 default if you like
            white = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.imwrite(tmp_bg_path, white)

    # Unique output filename
    job_id      = uuid.uuid4().hex
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    # Run processing (with progress bar)
    try:
        process_video(tmp_vid_path, tmp_bg_path, output_path)
    except Exception as e:
        os.unlink(tmp_vid_path)
        if bg_file:
            os.unlink(tmp_bg_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # Cleanup temp files
    os.unlink(tmp_vid_path)
    if bg_file:
        os.unlink(tmp_bg_path)

    # Stream back result
    def iterfile():
        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                yield chunk

    return StreamingResponse(iterfile(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000, log_level="info")
