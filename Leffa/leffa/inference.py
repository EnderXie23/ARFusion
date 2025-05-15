from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from leffa.pipeline import LeffaPipeline, OptimizedLeffaPipeline


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class LeffaInference(object):
    def __init__(
        self,
        model: nn.Module,
        use_fp16: bool = False,
        low_resolution: bool = False,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_fp16 and self.device == "cuda":
            self.model = model.half().to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.eval()

        self.pipe = OptimizedLeffaPipeline(model=self.model, use_dpm_solver=True, use_torch_compile=False)
        self.low_resolution = low_resolution

    def to_gpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = self.to_gpu(data)

        if self.low_resolution:
            print(f"[WARNING] Using low resolution for inference, original image has size {data['src_image'].shape[2:]}, lowered resolution is {data['src_image'].shape[2] // 2}x{data['src_image'].shape[3] // 2}.")
            new_size = (data["src_image"].shape[2] // 2, data["src_image"].shape[3] // 2)
            data["src_image"] = torch.nn.functional.interpolate(
                data["src_image"], size=new_size, mode="bilinear", align_corners=False
            )
            data["ref_image"] = torch.nn.functional.interpolate(
                data["ref_image"], size=new_size, mode="bilinear", align_corners=False
            )
            data["mask"] = torch.nn.functional.interpolate(
                data["mask"], size=new_size, mode="nearest"
            )
            data["densepose"] = torch.nn.functional.interpolate(
                data["densepose"], size=new_size, mode="nearest"
            )

        ref_acceleration = kwargs.get("ref_acceleration", self.low_resolution)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 2.5)
        seed = kwargs.get("seed", 42)
        repaint = kwargs.get("repaint", False)
        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        images = self.pipe(
            src_image=data["src_image"],
            ref_image=data["ref_image"],
            mask=data["mask"],
            densepose=data["densepose"],
            ref_acceleration=ref_acceleration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            repaint=repaint,
        )[0]

        # images = [pil_to_tensor(image) for image in images]
        # images = torch.stack(images)

        outputs = {}
        outputs["src_image"] = (data["src_image"] + 1.0) / 2.0
        outputs["ref_image"] = (data["ref_image"] + 1.0) / 2.0
        outputs["generated_image"] = images
        return outputs
