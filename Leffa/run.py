# python run.py --ref_acceleration --use_fp16 --low_resolution --step 10
import os
import numpy as np
from PIL import Image
import cv2
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image, resize_to_fixed_size
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

import argparse

class LeffaPredictor(object):
    def __init__(self, **args):
        self.mask_predictor = AutoMasker(
            densepose_path="/root/autodl-tmp/data/ckpts/densepose",
            schp_path="/root/autodl-tmp/data/ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="/root/autodl-tmp/data/ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="/root/autodl-tmp/data/ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="/root/autodl-tmp/data/ckpts/humanparsing/parsing_atr.onnx",
            lip_path="/root/autodl-tmp/data/ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="/root/autodl-tmp/data/ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="/root/autodl-tmp/data/ckpts/stable-diffusion-inpainting",
            pretrained_model="/root/autodl-tmp/data/ckpts/virtual_tryon.pth",
            # pretrained_model="../../StableVITON/ckpts/VITONHD.ckpt",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd, use_fp16=args.get("use_fp16", False), low_resolution=args.get("low_resolution", False))

        # vt_model_dc = LeffaModel(
        #     pretrained_model_name_or_path="/root/autodl-tmp/data/ckpts/stable-diffusion-inpainting",
        #     pretrained_model="/root/autodl-tmp/data/ckpts/virtual_tryon_dc.pth",
        #     dtype="float16",
        # )
        # self.vt_inference_dc = LeffaInference(model=vt_model_dc)

        # pt_model = LeffaModel(
        #     pretrained_model_name_or_path="/root/autodl-tmp/data/ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
        #     pretrained_model="/root/autodl-tmp/data/ckpts/pose_transfer.pth",
        #     dtype="float16",
        # )
        # self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False,
        low_resolution=False,
    ):
        # Open and resize the source image.
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # For virtual try-on, optionally preprocess the garment (reference) image.
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                # preprocess_garment_image returns a 768x1024 image.
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            # Otherwise, load the reference image.
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512) if not low_resolution else (192, 256)))
            keypoints = self.openpose(src_image.resize((384, 512) if not low_resolution else (192, 256)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024) if not low_resolution else (384, 512))
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        transform = LeffaTransform() if not low_resolution else LeffaTransform(height=512, width=384)
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            # elif vt_model_type == "dress_code":
            #     inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
            low_resolution=low_resolution,
        )
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,  # Pass through the new flag.
        )

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "pose_transfer",
            ref_acceleration,
            step,
            scale,
            seed,
        )
    
    def leffa_predict_stream(
        self,
        src_image,
        ref_image_path,
        ref_acceleration=False,
        step=10,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False,
        low_resolution=False,
    ):
        # Resize the source image.
        # src_image = resize_and_center(src_image, 768, 1024)
        src_image = resize_to_fixed_size(src_image, 768, 1024)

        # For virtual try-on, optionally preprocess the garment (reference) image.
        if preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                # preprocess_garment_image returns a 768x1024 image.
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            # Otherwise, load the reference image.
            ref_image = Image.open(ref_image_path)
            
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512) if not low_resolution else (192, 256)))
        keypoints = self.openpose(src_image.resize((384, 512) if not low_resolution else (192, 256)))
        if vt_model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        elif vt_model_type == "dress_code":
            mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024) if not low_resolution else (384, 512))

        if vt_model_type == "viton_hd":
            src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
            src_image_seg = Image.fromarray(src_image_seg_array)
            densepose = src_image_seg
        elif vt_model_type == "dress_code":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
            src_image_seg_array = src_image_iuv_array[:, :, 0:1]
            src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
            src_image_seg = Image.fromarray(src_image_seg_array)
            densepose = src_image_seg

        transform = LeffaTransform(height=512, width=384)
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if vt_model_type == "viton_hd":
            inference = self.vt_inference_hd
        # elif vt_model_type == "dress_code":
        #     inference = self.vt_inference_dc
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
            low_resolution=low_resolution,
        )
        gen_image = resize_to_fixed_size(output["generated_image"][0], 480, 640)
        return np.array(gen_image), np.array(mask), np.array(densepose)



if __name__ == "__main__":
    example_dir = "/root/autodl-tmp/data/ckpts/examples"
    person1_images = list_dir(f"{example_dir}/person1")
    person2_images = list_dir(f"{example_dir}/person2")
    garment_images = list_dir(f"{example_dir}/garment")

    argparser = argparse.ArgumentParser(description="Leffa Predictor")
    argparser.add_argument("--src_image", type=str, default=person1_images[0], help="Path to the source image")
    argparser.add_argument("--ref_image", type=str, default=garment_images[-1], help="Path to the reference image")
    # argparser.add_argument("--garment_image", type=str, default=garment_images[0], help="Path to the garment image")
    argparser.add_argument("--ref_acceleration", action="store_true", help="Enable reference acceleration")
    argparser.add_argument("--step", type=int, default=50, help="Number of inference steps")
    argparser.add_argument("--scale", type=float, default=2.5, help="Guidance scale")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed")
    argparser.add_argument("--vt_model_type", type=str, default="viton_hd", help="Virtual try-on model type")
    argparser.add_argument("--vt_garment_type", type=str, default="upper_body", help="Virtual try-on garment type")
    argparser.add_argument("--vt_repaint", action="store_true", help="Enable repainting")
    argparser.add_argument("--preprocess_garment", action="store_true", help="Enable garment preprocessing")
    argparser.add_argument("--output_dir", type=str, default="./output", help="Output directory for generated images")
    argparser.add_argument("--use_fp16", action="store_true", help="Use FP16 for inference")
    argparser.add_argument("--low_resolution", action="store_true", help="Use low resolution for inference")
    args = argparser.parse_args()

    leffa_predictor = LeffaPredictor(use_fp16=args.use_fp16, low_resolution=args.low_resolution)
    # leffa_predictor = LeffaPredictor(use_fp16=False, low_resolution=False)

    # Example usage
    src_image_path = args.src_image
    ref_image_path = args.ref_image
    ref_acceleration = args.ref_acceleration
    step = args.step
    scale = args.scale
    seed = args.seed
    vt_model_type = args.vt_model_type
    vt_garment_type = args.vt_garment_type
    vt_repaint = args.vt_repaint
    preprocess_garment = args.preprocess_garment
    # Call the virtual try-on prediction function
    # Record start time
    import time
    start_time = time.time()
    for _ in range(50):
        gen_image, mask, densepose = leffa_predictor.leffa_predict_vt(
            src_image_path,
            ref_image_path,
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment
        )
    # Record end time
    end_time = time.time()
    print(f"Time taken for 50 iterations: {end_time - start_time} seconds")
    print(f"Time taken for each iteration: {(end_time - start_time) / 50} seconds")

    # Save the generated image
    gen_image_pil = Image.fromarray(gen_image)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/generated_image.png"
    gen_image_pil.save(save_path)
    print(f"Generated image saved to {save_path}")

    # for _step in range(1, 16):
    #     gen_image, mask, densepose = leffa_predictor.leffa_predict_vt(
    #         src_image_path,
    #         ref_image_path,
    #         ref_acceleration,
    #         _step,
    #         scale,
    #         seed,
    #         vt_model_type,
    #         vt_garment_type,
    #         vt_repaint,
    #         preprocess_garment
    #     )
    #     gen_image_pil = Image.fromarray(gen_image)
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     save_path = f"{args.output_dir}/generated_{_step}.png"
    #     gen_image_pil.save(save_path)
    #     print(f"Generated image saved to {save_path}")
