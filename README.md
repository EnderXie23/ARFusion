# ARFusion: SkyAR & VITON

This is the repository for our CV Course Lab.

## Environment Setup
1. Clone the repository:
```bash
git clone https://github.com/EnderXie23/ARFusion.git
cd ARFusion
```
2. Create a virtual environment:
```bash
conda create -n arfusion python=3.10
conda activate arfusion
```
3. Install the required packages:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
```
4. Install `ffmpeg`:
```bash
conda install -c conda-forge ffmpeg
```

## Download the models:
The models checkpoints for SkyAR is placed at [Tsinghua Cloud - checkpoints_all.tar.gz](https://cloud.tsinghua.edu.cn/f/54b689b4e108453bbf70/), and the models checkpoints for VITON is placed at [Tsinghua Cloud - ckpts_filtered.tar.gz](https://cloud.tsinghua.edu.cn/f/eb6e02de759d43509e37/).

Download the first tar file and extract it to `ARFusion/ARFusion/SkyAR/`, and the second tar file and extract it to `ARFusion/ARFusion/Leffa/`.
```bash
cd SkyAR
wget https://cloud.tsinghua.edu.cn/f/54b689b4e108453bbf70/
tar -xvf checkpoints_all.tar.gz

cd ../Leffa
wget https://cloud.tsinghua.edu.cn/f/eb6e02de759d43509e37/
tar -xvf ckpts_filtered.tar.gz
```


## Run the video demo:
First of all, you shall fill in the `config.py` with your own input and output paths. The main parameters are:
- 'data_dir': The path to the input video, *relative to the `SkyAR` folder*.
- 'skybox': Available skyboxes are located at `SkyAR/skybox/`. You can also add your own skybox images.
- 'batch_size': The batch size for the model. If you have a powerful GPU, you can increase this value to speed up the processing.
- 'num_workers': The number of workers for the Leffa model. If you have a powerful GPU, you can increase this value to speed up the processing.
- 'ref_image': The reference garment image for the Leffa model. You can use any garment image, but it is recommended to use a white-background image for better results. The path is *relative to the `Leffa` folder*.
- 'output_dir': The path to the output video. The output video will be saved in this folder. The path is *relative to the `Leffa` folder*.

Then, run the demo with:
```bash
python video.py
```
The output video will be saved in the `output_dir` specified with the name `output_video.mp4`.


## Run streaming:
First, make sure to install `uvicorn` with `websockets` support. You can do this by running:
```bash
pip install "uvicorn[standard]"
# Or simply, run:
pip install -r requirements.txt
```

Then, run the server with:
```bash
uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

For a beatiful web-UI based access, simply click on the URL and open up the webpage in your browser.

Or for a software based access, on your local machine (also install `uvicon[standard]`), run:
```bash
python frontend.py
```
