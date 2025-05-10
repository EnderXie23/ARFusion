from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import asyncio
import utils
from skymagic_batch import SkyFilterBatched

app = FastAPI()

# Load once at import-time (no argparse here)
config_path = './config/test.json'
args        = utils.parse_config(config_path)
skyfilter   = SkyFilterBatched(args)
batch_size  = skyfilter.batch_size

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connection accepted")

    # Per-connection buffers
    frame_buffer      = []
    img_HD_prev_buf   = None

    try:
        loop = asyncio.get_running_loop()

        while True:
            # 1) receive raw bytes
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                 cv2.IMREAD_COLOR)
            frame_buffer.append(frame)

            # 2) wait until we have a full batch
            if len(frame_buffer) < batch_size:
                continue

            # 3) offload preprocessing to threadpool
            img_HD_batch_uint8 = await loop.run_in_executor(
                None,
                lambda: [skyfilter.cvtcolor_and_resize(f)
                         for f in frame_buffer]
            )
            img_HD_batch = [f.astype(np.float32)/255.0
                            for f in img_HD_batch_uint8]

            # 4) initialize / pad previous buffer
            if img_HD_prev_buf is None or len(img_HD_prev_buf) != batch_size:
                img_HD_prev_buf = img_HD_batch.copy()

            skyfilter.calibrate_skyengine()  # optional

            # 5) offload synthesis
            results = await loop.run_in_executor(
                None,
                skyfilter.synthesize_batch,
                img_HD_batch,
                img_HD_prev_buf
            )
            img_HD_prev_buf = img_HD_batch.copy()

            # 6) send each frame back
            # print("Sending back results…")
            for syn in results:
                syn_uint8 = (syn * 255).astype(np.uint8)
                syn_bgr  = cv2.cvtColor(syn_uint8,
                                        cv2.COLOR_RGB2BGR)
                _, enc = cv2.imencode('.jpg', syn_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                await ws.send_bytes(enc.tobytes())

            frame_buffer.clear()

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        await ws.close()
