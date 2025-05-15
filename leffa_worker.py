# leffa_worker.py
import os
import sys
import cv2
import numpy as np
from PIL import Image
from Leffa.run import LeffaPredictor

# 初始化全局 Leffa 实例（每个子进程各有一个）
leffa = None

def init_worker():
    global leffa
    # leffa = LeffaPredictor(use_fp16=True, low_resolution=True)
    leffa = LeffaPredictor()

def run_leffa(frame_bytes: bytes, ref_image_path: str) -> np.ndarray:
    global leffa

    # 1. decode bytes to image
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # 2. 运行 Leffa 推理
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
