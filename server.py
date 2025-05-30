import os, sys
os.environ["TQDM_DISABLE"] = "1" # Disable tqdm progress bar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Leffa')))
import leffa_worker

from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import asyncio
import json
import SkyAR.utils as skyar_utils
from SkyAR.skymagic_batch import SkyFilterBatched
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import threading

USE_LEFFA = False
os.chdir('SkyAR')
if USE_LEFFA:
    leffa_executor = ProcessPoolExecutor(max_workers=3, initializer=leffa_worker.init_worker)
    src_image_path = "/root/autodl-tmp/data/ckpts/examples/person1/01350_00.jpg"
    ref_image_path = "/root/autodl-tmp/data/ckpts/examples/garment/garment.png"
    
    def warmup_leffa():
        print("[WARMUP] Starting Leffa warmup...")
        img = cv2.imread(src_image_path)
        _, enc = cv2.imencode(".jpg", img)
        frame_bytes = enc.tobytes()
        try:
            fut = leffa_executor.submit(
                leffa_worker.run_leffa,
                frame_bytes,
                ref_image_path
            )
            _ = fut.result(timeout=60)
            print("[WARMUP] Leffa warmup done.")
        except Exception as e:
            print(f"[WARMUP ERROR] {e}")

    # Start the warmup in a separate thread
    threading.Thread(target=warmup_leffa, daemon=True).start()

# load a default config on startup
config = skyar_utils.parse_config('./config/test.json')
skyfilter = SkyFilterBatched(config)
batch_size = skyfilter.batch_size

# Mount the app on a html file
app = FastAPI()

@app.get("/")
async def get_index():
    # assumes static/index.html exists
    return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global config, skyfilter, batch_size

    await ws.accept()
    print("WebSocket connection accepted")

    try:
        # 1) **receive initial JSON configuration**
        msg = await ws.receive_text()
        new_cfg = json.loads(msg)
        print("Received new config:", new_cfg)
        # 2) **reinitialize our filter** with the new parameters
        args = skyar_utils.Struct(**new_cfg)
        skyfilter = SkyFilterBatched(args)
        batch_size = skyfilter.batch_size
        await ws.send_text("CONFIG_UPDATED")
    except Exception as e:
        print("Config error:", e.with_traceback(None))
        await ws.send_text("CONFIG_ERROR")
        await ws.close()
        return

    # Per-connection buffers
    frame_buffer    = []
    img_HD_prev_buf = None
    cali_batch = init_cali_batch = 4
    frame_num = 0
    per_frame_num = 100

    try:
        loop = asyncio.get_running_loop()
        while True:
            data = await ws.receive_bytes()
            frame_num += 1
            if USE_LEFFA and frame_num % per_frame_num == 0:
                try:
                    loop = asyncio.get_running_loop()
                    gen_image_rgb = await loop.run_in_executor(
                        leffa_executor,
                        leffa_worker.run_leffa,
                        data,  # raw bytes
                        ref_image_path
                    )
                    # Convert output to OpenCV BGR format
                    frame = cv2.cvtColor(gen_image_rgb, cv2.COLOR_RGB2BGR)

                except Exception as e:
                    print(f"LeffaPredictor error: {e}")
            else:
                # 1) decode bytes to image
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame)

            # print('buffer:', len(frame_buffer))
            if len(frame_buffer) < batch_size:
                continue

            # offload preprocessing & inference
            img_HD_batch_uint8 = await loop.run_in_executor(
                None,
                lambda: [skyfilter.cvtcolor_and_resize(f)
                         for f in frame_buffer]
            )
            img_HD_batch = [f.astype(np.float32)/255.0
                            for f in img_HD_batch_uint8]

            if img_HD_prev_buf is None or len(img_HD_prev_buf) != batch_size:
                img_HD_prev_buf = img_HD_batch.copy()

            if cali_batch == 0:
                skyfilter.calibrate_skyengine()  # optional
                cali_batch = init_cali_batch
            else:
                cali_batch -= 1

            results = await loop.run_in_executor(
                None,
                skyfilter.synthesize_batch,
                img_HD_batch,
                img_HD_prev_buf
            )
            img_HD_prev_buf = img_HD_batch.copy()

            # send back each processed frame
            for syn in results:
                syn_uint8 = (syn * 255).astype(np.uint8)
                syn_bgr  = cv2.cvtColor(syn_uint8,
                                        cv2.COLOR_RGB2BGR)
                _, enc = cv2.imencode(
                    '.jpg', syn_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                await ws.send_bytes(enc.tobytes())
            frame_buffer.clear()

    except Exception as e:
        print(f"WebSocket closed: {e.with_traceback(None)}")
    finally:
        try:
            await ws.close()
        except RuntimeError as e:
            print(f"Cancelling any future send.")
        print("Connection teardown complete")
