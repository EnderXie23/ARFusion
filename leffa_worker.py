# leffa_worker.py
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from Leffa.run import LeffaPredictor

class LeffaWorker:
    def __init__(self, gpu_id: int):
        """
        Initialize the Leffa worker with a specific GPU ID.
        :param gpu_id: The GPU ID to use for this worker.
        """
        self.gpu_id = gpu_id
        self.leffa = None  # Delayed init to be safe in subprocesses

    def setup(self):
        torch.cuda.set_device(self.gpu_id)
        print(f"[Leffa worker PID {os.getpid()}] → using GPU {self.gpu_id}")
        self.leffa = LeffaPredictor(use_fp16=True)

    def __call__(self, frame_bytes: bytes, ref_image_path: str) -> np.ndarray:
        if self.leffa is None:
            raise RuntimeError("LeffaPredictor not initialized. Call setup() first.")
        
        # 1. decode bytes to image
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # 2. do the Leffa inference
        gen_image, _, _ = self.leffa.leffa_predict_stream(
            src_image=pil_img,
            ref_image_path=ref_image_path,
            ref_acceleration=True,
            step=10,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            preprocess_garment=False
        )

        return np.array(gen_image)


leffa = None

def init_worker(gpu_id):
    global leffa
    torch.cuda.set_device(gpu_id)
    print(f"[PID {os.getpid()}] Using GPU {gpu_id}")
    leffa = LeffaPredictor(use_fp16=True)

def run_leffa(frame_bytes: bytes, ref_image_path: str) -> np.ndarray:
    global leffa

    # 1. decode bytes to image
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # 2. do the Leffa inference
    gen_image, _, _ = leffa.leffa_predict_stream(
        src_image=pil_img,
        ref_image_path=ref_image_path,
        ref_acceleration=True,
        step=10,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False
    )

    return np.array(gen_image)  # RGB 格式的 numpy array
