#!/usr/bin/env python3
import argparse
import asyncio
import logging
import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

async def stream_video(
    ws_url: str,
    width: int,
    height: int,
    quality: int,
    ping_interval: float,
    ping_timeout: float,
    open_timeout: float,
):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open webcam.")
        return

    stop_event = asyncio.Event()

    async def send_loop(ws):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Webcam read failed; stopping send.")
                stop_event.set()
                break

            frame = cv2.resize(frame, (width, height))
            ok, buf = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                logging.warning("Failed to encode frame; skipping.")
                await asyncio.sleep(0)
                continue

            try:
                await ws.send(buf.tobytes())
            except Exception as e:
                logging.error(f"Send failed: {e}")
                stop_event.set()
                break

            # yield to other tasks
            await asyncio.sleep(0)

    async def recv_loop(ws):
        win = "Processed (WebSocket)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)    # create the window

        while not stop_event.is_set():
            try:
                data = await ws.recv()
            except ConnectionClosedOK:
                logging.info("Server closed connection cleanly.")
                stop_event.set()
                break
            except ConnectionClosedError as e:
                logging.error(f"Connection closed with error: {e}")
                stop_event.set()
                break

            proc = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            # cv2.imshow(win, proc)
            if proc is None:
                logging.warning("Malformed frame received; skipping display.")
                await asyncio.sleep(0)
                continue

            cv2.imshow("Processed (WebSocket)", proc)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("'q' pressed; exiting.")
                stop_event.set()
                break

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("Window closed by user; exiting.")
                stop_event.set()
                break

            await asyncio.sleep(0)

    try:
        async with websockets.connect(
            ws_url,
            max_size=None,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            open_timeout=open_timeout,
        ) as ws:
            logging.info(f"Connected to {ws_url}")
            # run both loops concurrently
            await asyncio.gather(
                send_loop(ws),
                recv_loop(ws),
            )
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleaned up webcam and windows.")

def main():
    parser = argparse.ArgumentParser(description="WebSocket video client")
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws",
                        help="WebSocket server URL")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--quality", type=int, default=100,
                        help="JPEG quality (0-100)")
    parser.add_argument("--ping-interval", type=float, default=20.0,
                        help="WebSocket ping interval (s)")
    parser.add_argument("--ping-timeout", type=float, default=20.0,
                        help="WebSocket ping timeout (s)")
    parser.add_argument("--open-timeout", type=float, default=10.0,
                        help="Handshake timeout (s)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s"
    )

    asyncio.run(stream_video(
        ws_url=args.url,
        width=args.width,
        height=args.height,
        quality=args.quality,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        open_timeout=args.open_timeout,
    ))

if __name__ == "__main__":
    main()
