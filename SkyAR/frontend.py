#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import time
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, BooleanVar, StringVar
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

ckpt_lookup = {
    "eff":            "./checkpoints",
    "bisenetv2":      "./checkpoints_bisenet",
    "alexnet":        "./checkpoints_alex",
    "coord_resnet101":"./checkpoints101",
    "coord_resnet50": "./checkpoints_G_coord_resnet50",
}
skyboxes = [
    "galaxy.jpg",
    "cloudy.jpg",
    "sunny.jpg",
    "sunset.jpg",
    "supermoon.jpg",
    "jupiter.jpg",
    "district9ship.jpg",
    "floatingcastle.jpg",
    "thunderstorm.mp4",
]

def get_config(defaults):
    """
    Pop up a simple Tkinter form for the user to tweak all parameters,
    then return a dict.
    """
    cfg = {}

    root = tk.Tk()
    root.title("SkyFilter Configuration")
    root.geometry("330x500")
    root.configure(bg="#f5f5f5")

    style = ttk.Style(root)
    style.theme_use("clam")   # available default theme
    style.configure("TLabelframe", background="#f5f5f5", font=("Arial", 11, "bold"))
    style.configure("TLabel",       background="#f5f5f5", font=("Arial", 10))
    style.configure("TCombobox",    font=("Arial", 10))
    style.configure("TEntry",       font=("Arial", 10))
    style.configure("TCheckbutton", background="#f5f5f5", font=("Arial", 10))
    style.configure("TButton",      font=("Arial", 10, "bold"), padding=6)

    # ─── Group fields into LabelFrames ─────────────────────────────────────
    model_frame  = ttk.LabelFrame(root, text="Model", padding=8)
    res_frame    = ttk.LabelFrame(root, text="Resolution", padding=8)
    filter_frame = ttk.LabelFrame(root, text="Filtering", padding=8)
    stream_frame = ttk.LabelFrame(root, text="Stream", padding=8)

    model_frame. grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    res_frame.   grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    filter_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    stream_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

    # define fields: (key, label, widget_type, options_or_default)
    fields = [
        ("net_G",       "Generator Network",     "combo",  list(ckpt_lookup.keys()), model_frame),
        ("skybox",      "Skybox Name",           "combo",  skyboxes, model_frame),

        ("in_size_w",   "Input Width",           "entry",  defaults["in_size_w"], res_frame),
        ("in_size_h",   "Input Height",          "entry",  defaults["in_size_h"], res_frame),
        ("out_size_w",  "Output Width",          "entry",  defaults["out_size_w"], res_frame),
        ("out_size_h",  "Output Height",         "entry",  defaults["out_size_h"], res_frame),

        ("skybox_center_crop","Center Crop",      "entry",  defaults["skybox_center_crop"], filter_frame),
        ("auto_light_matching","Auto Light Matching","check", defaults["auto_light_matching"], filter_frame),
        ("relighting_factor","Relighting Factor", "entry",  defaults["relighting_factor"], filter_frame),
        ("recoloring_factor","Recoloring Factor", "entry",  defaults["recoloring_factor"], filter_frame),
        ("halo_effect",       "Halo Effect",       "check",  defaults["halo_effect"], filter_frame),

        ("width",       "Stream Width",          "entry",  defaults["width"], stream_frame),
        ("height",      "Stream Height",         "entry",  defaults["height"], stream_frame),
        ("quality",     "JPEG Quality",          "entry",  defaults["quality"], stream_frame),
    ]

    vars = {}
    for i, (key, label, kind, opt, _) in enumerate(fields):
        tk.Label(root, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
        if kind == "combo":
            v = StringVar(value=opt[0])
            cb = ttk.Combobox(root, textvariable=v, values=opt, state="readonly")
            cb.grid(row=i, column=1, padx=5, pady=2)
            vars[key] = v
        elif kind == "check":
            v = BooleanVar(value=opt)
            cb = tk.Checkbutton(root, variable=v)
            cb.grid(row=i, column=1, padx=5, pady=2, sticky="w")
            vars[key] = v
        else:
            v = StringVar(value=str(opt))
            # make ckptdir entry read-only
            if kind == "entry-ro":
                ent = tk.Entry(root, textvariable=v, state="readonly")
            else:
                ent = tk.Entry(root, textvariable=v)
            ent.grid(row=i, column=1, padx=5, pady=2)
            vars[key] = v

    # Start button
    def on_start():
        try:
            for key in vars:
                val = vars[key].get()
                # cast booleans, ints, floats
                if key in ("auto_light_matching", "halo_effect"):
                    cfg[key] = bool(val)
                elif key in ("in_size_w","in_size_h","out_size_w","out_size_h","width","height","quality"):
                    cfg[key] = int(val)
                elif key in ("skybox_center_crop","relighting_factor","recoloring_factor"):
                    cfg[key] = float(val)
                else:
                    cfg[key] = val
            cfg["input_mode"] = "video"
            cfg["datadir"] = "./test_videos/test.mp4"
            cfg["output_dir"] = "./jpg_output"
            cfg["save_jpgs"] = False
            cfg["ckptdir"] = ckpt_lookup[cfg["net_G"]]
        except Exception as e:
            tk.messagebox.showerror("Invalid input", str(e))
            return
        root.destroy()
    
    def on_cancel():
        root.destroy()
        return None

    ctrl = ttk.Frame(root, padding=10, relief="raised")
    ctrl.grid(row=14, column=0, columnspan=2, sticky="ew")
    ttk.Button(ctrl, text="Start",       command=on_start).grid(row=0, column=0, padx=5)
    ttk.Button(ctrl, text="Cancel",      command=on_cancel).grid(row=0, column=1, padx=5)

    root.mainloop()
    return cfg

async def stream_video(ws_url, config, ping_interval, ping_timeout, open_timeout):
    main_loop = asyncio.get_running_loop()
    restart_event = asyncio.Event()
    shutdown_event = asyncio.Event()
    ping_ms = None

    while not shutdown_event.is_set():
        cap = cv2.VideoCapture(0)
        win = "Processed (WebSocket)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # initial “please wait” screen
        h, w = config["height"], config["width"]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(blank,
                    "Waiting for server response...",
                    (w//10, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA)
        cv2.imshow(win, blank)
        cv2.waitKey(1)

        # ─── “Re-configure” button geometry & callback ──────────────────────
        btn_w, btn_h = 140, 30
        mx, my = 10, 10
        btn_rect = {}
        def on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            rx, ry = btn_rect.get("x", -1), btn_rect.get("y", -1)
            if rx <= x <= rx + btn_w and ry <= y <= ry + btn_h:
                new_cfg = get_config(config)
                if not new_cfg or not new_cfg.get("net_G"):
                    logging.info("User cancelled configuration.")
                    return
                config.update(new_cfg)
                async def reconfig_and_restart():
                    await ws.send(json.dumps(config))
                    restart_event.set()
                    await ws.close()
                asyncio.run_coroutine_threadsafe(reconfig_and_restart(), main_loop)

        cv2.setMouseCallback(win, on_mouse)

        try:
            async with websockets.connect(
                ws_url,
                max_size=None,
                ping_interval=None,
                ping_timeout=ping_timeout,
                open_timeout=open_timeout,
            ) as ws:
                # Initial handshake
                await ws.send(json.dumps(config))
                ack = await ws.recv()
                if ack != "CONFIG_UPDATED":
                    logging.error("Bad CONFIG reply, stopping.")
                    return

                # ─── ping loop ─────────────────────────────────────────────────
                async def ping_loop():
                    nonlocal ping_ms
                    while True:
                        await asyncio.sleep(ping_interval)
                        try:
                            t0 = asyncio.get_event_loop().time()
                            pong = await ws.ping()
                            await pong
                            ping_ms = (asyncio.get_event_loop().time() - t0) * 1000
                        except:
                            break

                # ─── send loop ─────────────────────────────────────────────────
                async def send_loop():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (config["width"], config["height"]))
                        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), config["quality"]])
                        if not ok:
                            continue
                        try:
                            await ws.send(buf.tobytes())
                        except ConnectionClosedOK:
                            logging.info("WebSocket closed.")
                            return
                        except ConnectionClosedError:
                            logging.error("WebSocket error on send.")
                            return

                # ─── recv loop (with UI pumping) ───────────────────────────────
                async def recv_loop():
                    while True:
                        try:
                            # try to recv, but time out quickly so we can pump UI
                            data = await ws.recv()
                        except asyncio.TimeoutError:
                            # no data yet → pump GUI events
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                shutdown_event.set()
                                return
                            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                                shutdown_event.set()
                                return
                            continue
                        except ConnectionClosedOK:
                            logging.info("WebSocket closed.")
                            return
                        except ConnectionClosedError:
                            logging.error("WebSocket error on receive.")
                            return

                        # got a frame!
                        if not isinstance(data, (bytes, bytearray)):
                            continue
                        proc = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if proc is None:
                            continue

                        # draw “Re-configure” button
                        h_p, w_p = proc.shape[:2]
                        x0 = w_p - btn_w - mx
                        y0 = my
                        btn_rect.update({"x": x0, "y": y0})
                        cv2.rectangle(proc, (x0, y0), (x0 + btn_w, y0 + btn_h), (50, 50, 50), -1)
                        cv2.rectangle(proc, (x0, y0), (x0 + btn_w, y0 + btn_h), (200, 200, 200), 1)
                        cv2.putText(proc, "Re-configure", (x0 + 8, y0 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # overlay ping time
                        if ping_ms is not None:
                            cv2.putText(proc, f"Ping: {ping_ms:.1f}ms", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.imshow(win, proc)

                        # pump GUI after each frame
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            shutdown_event.set()
                            return
                        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            shutdown_event.set()
                            return

                # ─── run all three loops until one completes ───────────────────
                tasks = [
                    asyncio.create_task(ping_loop()),
                    asyncio.create_task(send_loop()),
                    asyncio.create_task(recv_loop()),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()

                # if we triggered a reconfigure, loop again
                if restart_event.is_set() and not shutdown_event.is_set():
                    restart_event.clear()
                    continue
                else:
                    break

        except asyncio.TimeoutError:
            logging.error("WebSocket handshake timed out. Retrying in 3s...")
            await asyncio.sleep(3)

        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            return

        finally:
            cap.release()
            try:
                cv2.destroyWindow(win)
            except:
                pass

    logging.info("Streaming stopped.")


def main():
    parser = argparse.ArgumentParser(description="WebSocket video client")
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws",
                        help="WebSocket server URL")
    parser.add_argument("--ping-interval", type=float, default=1.0,
                        help="Ping interval (s)")
    parser.add_argument("--ping-timeout", type=float, default=10.0,
                        help="Ping timeout (s)")
    parser.add_argument("--open-timeout", type=float, default=10.0,
                        help="Handshake timeout (s)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s"
    )

    # default config dict
    defaults = {
      "net_G": "eff",
      "ckptdir": ckpt_lookup["eff"],
      "input_mode": "video",
      "datadir": "./test_videos/test.mp4",
      "skybox": "galaxy.jpg",
      "in_size_w": 384,
      "in_size_h": 384,
      "out_size_w": 845,
      "out_size_h": 480,
      "skybox_center_crop": 0.5,
      "auto_light_matching": False,
      "relighting_factor": 0.8,
      "recoloring_factor": 0.5,
      "halo_effect": True,
      "output_dir": "./jpg_output",
      "save_jpgs": False,
      # local params:
      "width": args.width if hasattr(args, "width") else 640,
      "height": args.height if hasattr(args, "height") else 480,
      "quality": 100,
    }

    # pop-up panel
    config = get_config(defaults)
    if config is None or not config.get("net_G"):
        logging.info("User cancelled configuration.")
        return

    asyncio.run(stream_video(
        ws_url=args.url,
        config=config,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        open_timeout=args.open_timeout,
    ))

if __name__ == "__main__":
    main()
