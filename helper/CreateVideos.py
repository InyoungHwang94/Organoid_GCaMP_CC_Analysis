"""
===============================================================================
CreateVideos.py — Suite2p Calcium Imaging Video Generator
Created : 2026-03-24
Last Modified : 2026-03-24

Purpose
-------
Generates annotated MP4 videos from Suite2p registered binary data (data.bin),
rendering each frame as a green-channel calcium fluorescence image with an
overlaid scale bar and frame counter.

Pipeline Position
-----------------
Runs after  : Suite2p segmentation and signal extraction
Runs before : (standalone visualisation — not part of the analysis pipeline)

Notes
-----
- Iterates over all subfolders in the given root path, expecting each to
  contain a plane0/ directory with Suite2p output.
- Temporal binning (temporal_bin > 1) averages consecutive frames to reduce
  shot noise at the cost of temporal resolution.
- Output video is saved alongside the Suite2p data in plane0/.

===============================================================================
"""

from datetime import datetime
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_suite2p_data(suite2p_path):
    """Load Suite2p data from the specified path."""
    ops = np.load(os.path.join(suite2p_path, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(suite2p_path, 'stat.npy'), allow_pickle=True)
    F = np.load(os.path.join(suite2p_path, 'F.npy'))
    Fneu = np.load(os.path.join(suite2p_path, 'Fneu.npy'))

    # Load registered binary data if available
    reg_file = os.path.join(suite2p_path, 'data.bin')
    if os.path.exists(reg_file):
        reg_data = np.fromfile(reg_file, dtype=np.int16)
        n_frames = ops['nframes']
        Ly, Lx = ops['Ly'], ops['Lx']
        reg_data = reg_data.reshape((n_frames, Ly, Lx))
    else:
        reg_data = None

    return ops, stat, F, Fneu, reg_data


def normalize_frame(frame, percentile_min=1, percentile_max=99):
    """Normalize frame to 0-255 range using percentile clipping."""
    vmin = np.percentile(frame, percentile_min)
    vmax = np.percentile(frame, percentile_max)
    frame_norm = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
    return (frame_norm * 255).astype(np.uint8)


def add_scale_bar(frame, ops, pixel_size_um=1.0, scale_bar_um=50):
    """Add a white scale bar with white text to the frame at bottom-right corner."""
    scale_bar_pixels = int(scale_bar_um / pixel_size_um)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    margin = 20
    bar_y = frame.shape[0] - margin
    bar_x2 = frame.shape[1] - margin
    bar_x1 = bar_x2 - scale_bar_pixels

    bar_thickness = 4
    bar_color = (255, 255, 255)

    for i in range(bar_thickness):
        draw.line(
            [(bar_x1, bar_y + i - bar_thickness // 2),
             (bar_x2, bar_y + i - bar_thickness // 2)],
            fill=bar_color, width=1
        )

    text = f"{scale_bar_um} um"
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = bar_x1 + (scale_bar_pixels - text_width) // 2
    text_y = bar_y - text_height - 12

    draw.text((text_x, text_y), text, fill=bar_color, font=font)

    return np.array(pil_img)


def add_timestamp(frame, frame_num, total_frames, fps=30):
    """Add frame counter in white text at the bottom centre of the frame."""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    text = f"Frame {frame_num}/{total_frames}"

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    color = (255, 255, 255)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    text_x = (frame.shape[1] - text_width) // 2
    text_y = frame.shape[0] - 45

    draw.text((text_x, text_y), text, fill=color, font=font)

    return np.array(pil_img)


def create_video_from_suite2p(suite2p_path,
                               fps=30,
                               pixel_size_um=1.0,
                               scale_bar_um=50,
                               start_frame=0,
                               end_frame=None,
                               codec='mp4v',
                               temporal_bin=1):
    """
    Create a video from Suite2p data with cells flashing in green.

    Parameters
    ----------
    suite2p_path : str
        Root folder containing per-recording subfolders, each with a plane0/
        directory holding Suite2p output (ops.npy, stat.npy, F.npy, data.bin).
    fps : float
        Frames per second of the output video.
    pixel_size_um : float
        Physical size of one pixel in micrometres (used for scale bar).
    scale_bar_um : float
        Length of the scale bar in micrometres.
    start_frame : int
        First frame index to include (0-based).
    end_frame : int or None
        Last frame index (exclusive).  None = all frames.
    codec : str
        FourCC codec string ('mp4v' for .mp4, 'XVID' for .avi).
    temporal_bin : int
        Number of frames to average per output frame (1 = no averaging).
    """
    folder_path = suite2p_path
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        plane0_path = os.path.join(subfolder, 'plane0')
        if not os.path.exists(plane0_path):
            print(f"Suite2p folder not found in {subfolder}, skipping...")
            continue

        print("Loading Suite2p data...")
        ops, stat, F, Fneu, reg_data = load_suite2p_data(plane0_path)

        if reg_data is None:
            print("Error: Could not find registered binary data (data.bin)")
            return

        n_frames = reg_data.shape[0]
        _end_frame = min(end_frame, n_frames) if end_frame is not None else n_frames

        n_frames_to_process = _end_frame - start_frame
        n_output_frames = n_frames_to_process // temporal_bin

        print(f"Creating video with {n_output_frames} frames at {fps} fps...")
        print(f"Temporal binning: averaging every {temporal_bin} frames")
        print(f"Image size: {ops['Ly']} x {ops['Lx']} pixels")

        folder_name = os.path.basename(subfolder)
        date_str = datetime.now().strftime("%Y%m%d")
        output_file_name = (
            f"{date_str}_{folder_name}_CalciumTransients_"
            f"Frames{start_frame}-{_end_frame}.mp4"
        )
        output_path = os.path.join(plane0_path, output_file_name)

        text_area_height = 50
        output_height = ops['Ly'] + text_area_height
        output_width = ops['Lx']

        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height), isColor=True)

        for bin_idx in range(n_output_frames):
            if bin_idx % 100 == 0:
                print(f"Processing frame {bin_idx}/{n_output_frames}...")

            start_idx = start_frame + bin_idx * temporal_bin
            end_idx = start_idx + temporal_bin

            frames_to_avg = reg_data[start_idx:end_idx]
            frame = np.mean(frames_to_avg, axis=0).astype(np.int16)

            frame_norm = normalize_frame(frame)

            frame_rgb = np.zeros((frame_norm.shape[0], frame_norm.shape[1], 3), dtype=np.uint8)
            frame_rgb[:, :, 1] = frame_norm          # green channel
            frame_rgb[:, :, 0] = frame_norm // 3     # slight red
            frame_rgb[:, :, 2] = frame_norm // 3     # slight blue

            frame_rgb = add_scale_bar(frame_rgb, ops, pixel_size_um, scale_bar_um)

            extended_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            extended_frame[:ops['Ly'], :, :] = frame_rgb

            display_frame = start_idx + temporal_bin // 2 - start_frame + 1
            extended_frame = add_timestamp(extended_frame, display_frame, n_frames_to_process, fps)

            out.write(extended_frame)

        out.release()
        print(f"Video saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    suite2p_path = r"F:\inyoung\B3_Dup_Org3_3x_GZ-001"

    create_video_from_suite2p(
        suite2p_path=suite2p_path,
        fps=15.11,
        pixel_size_um=0.364707528427891,
        scale_bar_um=50,
        start_frame=1,
        end_frame=2000,
        codec='mp4v',
        temporal_bin=3,
    )
