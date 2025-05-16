from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import asyncio
import json
import utils
from skymagic_batch import SkyFilterBatched
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Mount the app on a html file
@app.get("/")
async def get_index():
    # assumes static/index.html exists
    return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")

# load a default config on startup
config = utils.parse_config('./config/test.json')
skyfilter = SkyFilterBatched(config)
batch_size = skyfilter.batch_size

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
        args = utils.Struct(**new_cfg)
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
    cali_batch = init_cali_batch = 2

    try:
        loop = asyncio.get_running_loop()
        while True:
            # receive raw bytes (the actual video frames)
            data = await ws.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )
            frame_buffer.append(frame)

            # only process once we have a full batch
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
