# ARFusion: SkyAR & VITON

This is the repository for our CV Course Lab.

## Run streaming:
First, make sure to install `uvicorn` with `websockets` support. You can do this by running:
```bash
pip install "uvicorn[standard]"
# Or simply, run:
pip install -r requirements.txt
```

Then, run the server with (`PYTHONWARNINGS="ignore"` is optional):
```bash
PYTHONWARNINGS="ignore" uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

On your local machine (also install `uvicon[standard]`), run:
```bash
python frontend.py
```
