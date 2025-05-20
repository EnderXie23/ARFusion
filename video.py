#!/usr/bin/env python3
import os, sys
leffa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Leffa'))
sys.path.insert(0, leffa_path)

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

import json
import cv2
import shutil
import numpy as np
from types import SimpleNamespace
from Leffa.run import LeffaPredictor
from SkyAR.skymagic_batch import SkyFilterBatched
from tqdm.auto import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading

# Path to your unified JSON config
CONFIG_PATH = '../config.json'

os.chdir('SkyAR')

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    # Load unified config
    cfg = load_config(CONFIG_PATH)
    skyar_cfg_dict = cfg['skyar']
    leffa_cfg = cfg['leffa']
    video_cfg = cfg['video']

    # Convert SkyAR part into a config object
    skyar_cfg = SimpleNamespace(**skyar_cfg_dict)

    # Prepare output directories
    sky_frames_dir = os.path.join(video_cfg['output_dir'], 'sky_frames')
    leffa_frames_dir = os.path.join(video_cfg['output_dir'], 'leffa_frames')
    # Remove existing directories and files if they exist
    if os.path.exists(sky_frames_dir):
        print(f"Removing existing directory: {sky_frames_dir}")
        shutil.rmtree(sky_frames_dir)
    if os.path.exists(leffa_frames_dir):
        print(f"Removing existing directory: {leffa_frames_dir}")
        shutil.rmtree(leffa_frames_dir)
    os.makedirs(sky_frames_dir, exist_ok=True)
    os.makedirs(leffa_frames_dir, exist_ok=True)

    # 1) Run SkyAR on the input video
    print('Running SkyAR on video:', skyar_cfg.datadir)
    sf = SkyFilterBatched(skyar_cfg)
    sf.run()

    perform_leffa = leffa_cfg.get('perform', True)
    if perform_leffa:
        # Take demo.mp4 apart into frames
        cap = cv2.VideoCapture("demo.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_files = []
        while True:
            ret, frame = cap.read()
            # Store the frame as a file
            if not ret:
                break
            frame_files.append(f"frame_{len(frame_files):06d}.png")
            cv2.imwrite(os.path.join(sky_frames_dir, frame_files[-1]), frame)
        cap.release()

        # 2) Initialize Leffa predictor
        num_workers = video_cfg.get('num_workers', 2)
        gpu_ids = video_cfg.get('gpu_ids', [0])
        print(f'Initializing {num_workers} LeffaPredictor instances...')
        predictors = []
        for i in range(num_workers):
            # Pin each predictor to a different GPU if provided
            if gpu_ids:
                import torch
                gpu_id = gpu_ids[i % len(gpu_ids)]
                torch.cuda.set_device(gpu_id)
            predictors.append(
                LeffaPredictor(
                    use_fp16=leffa_cfg['use_fp16'],
                    low_resolution=leffa_cfg['low_resolution']
                )
            )

        thread_predictor_map = {}
        map_lock = threading.Lock()

        # 3) Multithreaded frame processing
        frame_files = sorted(
            f for f in os.listdir(sky_frames_dir)
            if f.lower().endswith(('.jpg', '.png'))
        )
        total_frames = len(frame_files)
        print(f'Processing {total_frames} frames across {num_workers} workers...')

        def process_frame(item):
            idx, fname = item
            # Select predictor for this thread
            tid = threading.get_ident()
            with map_lock:
                if tid not in thread_predictor_map:
                    thread_predictor_map[tid] = len(thread_predictor_map)
                pred_idx = thread_predictor_map[tid] % num_workers
            predictor = predictors[pred_idx]

            frame_path = os.path.join(sky_frames_dir, fname)
            gen_image, mask, densepose = predictor.leffa_predict_vt(
                frame_path,
                video_cfg['ref_image'],
                False,
                leffa_cfg['step'],
                leffa_cfg['scale'],
                leffa_cfg['seed'],
                leffa_cfg['vt_model_type'],
                leffa_cfg['vt_garment_type'],
                leffa_cfg['vt_repaint'],
                leffa_cfg['preprocess_garment']
            )
            out_path = os.path.join(leffa_frames_dir, f"{idx:06d}.png")
            print(f"Saving generated image {idx} to {out_path} using worker {pred_idx}...")
            if isinstance(gen_image, np.ndarray):
                cv2.imwrite(out_path, cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
            else:
                gen_image.save(out_path)
            return idx

        # Launch threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in executor.map(process_frame, enumerate(frame_files)):
                pass

        # 4) Assemble the final output video
        print('Assembling final video...')
        first_frame = cv2.imread(os.path.join(leffa_frames_dir, f"{0:06d}.png"))
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out_path = os.path.join(video_cfg['output_dir'], 'output_video_nosound.mp4')
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

        for idx in range(len(frame_files)):
            frame = cv2.imread(os.path.join(leffa_frames_dir, f"{idx:06d}.png"))
            writer.write(frame)
        writer.release()
    else: # if not perform_leffa
        video_out_path = os.path.join(video_cfg['output_dir'], 'output_video_nosound.mp4')
        shutil.copyfile("demo.mp4", video_out_path)
    print('Saved output video (no sound):', video_out_path)

    # 5) copy audio from original video to ouput
    command = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", video_out_path,            # video (no audio)
        "-i", skyar_cfg_dict["datadir"], # audio source
        "-c", "copy",                    # copy both video and audio without re-encoding
        "-map", "0:v:0",                 # take video from output_video
        "-map", "1:a:0",                 # take audio from test.mp4
        video_out_path.replace("_nosound", "")
    ]

    subprocess.run(command, check=True, capture_output=True)
    os.remove(video_out_path)
    print('Saved output video (with sound):', video_out_path.replace("_nosound", ""))


if __name__ == '__main__':
    main()
