import inspect
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image, ImageFilter
from rectified_flow import RectifiedFlow


class RectifiedFlowScheduler:
    def __init__(self, config: dict, velocity_field: nn.Module, data_shape: tuple,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        Adapted for stable-diffusion-inpainting.

        Args:
            config (dict): Scheduler configuration (e.g., from DDIM inpainting config).
            velocity_field (nn.Module): The network (usually UNet) used as the velocity field.
            data_shape (tuple): Expected shape of the full noisy sample (e.g., (12, 1024, 768)).
            device (torch.device): Device to run on.
            dtype (torch.dtype): Data type for computation.
        """
        self.config = config = {
            "_class_name": "DDIMScheduler",
            "_diffusers_version": "0.6.0.dev0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "steps_offset": 1,
            "trained_betas": None,
            "skip_prk_steps": True
        }
        self.velocity_field = velocity_field
        self.data_shape = data_shape  # e.g. (12, 1024, 768)
        self.device = device
        self.dtype = dtype
        self.order = 1  # used for progress updates in the pipeline
        
        # A blending factor for rectification (tunable hyperparameter)
        self.rectification_weight = 0.5
        
        # Diffusion schedule parameters from config
        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        beta_start = config.get("beta_start", 0.00085)
        beta_end = config.get("beta_end", 0.012)
        # Create a linear beta schedule (could also use config['trained_betas'] if available)
        self.betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps,
                                    dtype=self.dtype, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Noise scaling factor for initial noise (remains for compatibility)
        self.init_noise_sigma = 1.0

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps_inference = num_inference_steps
        # Create a linear spacing between num_train_timesteps and 0.
        self.timesteps = torch.linspace(self.num_train_timesteps, 0, steps=num_inference_steps).to(self.device)

    def _initialize_alphas_cumprod(self):
        # Compute a linear schedule for betas. If you use a "scaled_linear" schedule, you might adjust here.
        betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_train_timesteps, device=self.device, dtype=self.dtype)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def scale_model_input(self, latent_model_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Scales the latent model input based on the current timestep.

        Args:
            latent_model_input (torch.Tensor): The latent input to be scaled.
            t (torch.Tensor): The current timestep (assumed to be a scalar tensor).

        Returns:
            torch.Tensor: The scaled latent input.
        """
        if self.alphas_cumprod is None:
            self._initialize_alphas_cumprod()

        # Since our timesteps are generated from torch.linspace(num_train_timesteps, 0, steps=...) they are floats.
        # We convert the current timestep to an integer index into our precomputed alpha cumprod.
        t_index = int(torch.round(t).item())
        # Clamp index in case rounding falls outside the valid range.
        t_index = max(0, min(t_index, self.num_train_timesteps - 1))
        
        # Obtain the scaling factor: sqrt(alpha_cumprod[t_index]).
        scaling_factor = torch.sqrt(self.alphas_cumprod[t_index]).to(latent_model_input.device)
        
        # Scale the latent model input.
        return latent_model_input / scaling_factor
    
    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latent_model_input: torch.Tensor, **kwargs):
        """
        Performs a single update step for rectified flow in the reverse diffusion process.
        """
        # The first sample is taken as the unconditional input and the second as the conditional.
        latent_uncond, latent_cond = latent_model_input.chunk(2, dim=0)
        
        # Extract the first 4 channels from each branch to be used as latent noise prediction cues.
        # (This assumes that your concatenation layout is known and that the latent cues reside in these channels.)
        latent_uncond_noise = latent_uncond[:, :4]
        latent_cond_noise = latent_cond[:, :4]
        
        # Set or retrieve a guidance scale factor (this could be defined during initialization).
        guidance_scale = getattr(self, "rectification_weight", 1.0)
        
        # Rectify noise_pred by adding the weighted difference between conditional and unconditional cues.
        # This is similar in spirit to classifier-free guidance.
        refined_noise_pred = noise_pred + guidance_scale * (latent_cond_noise - latent_uncond_noise)

        new_latent = refined_noise_pred  # shape should be [1, 4, 128, 96]
        return (new_latent,)
    

class LeffaPipeline(object):
    def __init__(
        self,
        model,
        device="cuda",
        use_rectified_flow=False,
    ):
        self.vae = model.vae
        self.unet_encoder = model.unet_encoder
        self.unet = model.unet
        self.device = device

        if use_rectified_flow:
            self.noise_scheduler = RectifiedFlowScheduler(
                config=None,
                velocity_field=self.unet,  # using the same UNet as the velocity field
                data_shape=(12, 1024, 768),
                device=device,
                dtype=self.vae.dtype
            )
        else:
            self.noise_scheduler = model.noise_scheduler

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        src_image,
        ref_image,
        mask,
        densepose,
        ref_acceleration=False,
        num_inference_steps=50,
        do_classifier_free_guidance=True,
        guidance_scale=2.5,
        generator=None,
        eta=1.0,
        repaint=False,  # used for virtual try-on
        **kwargs,
    ):
        with torch.amp.autocast("cuda", enabled=True):
            src_image = src_image.to(device=self.vae.device, dtype=self.vae.dtype)
            ref_image = ref_image.to(device=self.vae.device, dtype=self.vae.dtype)
            mask = mask.to(device=self.vae.device, dtype=self.vae.dtype)
            densepose = densepose.to(device=self.vae.device, dtype=self.vae.dtype)
            masked_image = src_image * (mask < 0.5)

            # 1. VAE encoding
            with torch.no_grad():
                # src_image_latent = self.vae.encode(src_image).latent_dist.sample()
                masked_image_latent = self.vae.encode(masked_image).latent_dist.sample()
                ref_image_latent = self.vae.encode(ref_image).latent_dist.sample()
            # src_image_latent = src_image_latent * self.vae.config.scaling_factor
            masked_image_latent = masked_image_latent * self.vae.config.scaling_factor
            ref_image_latent = ref_image_latent * self.vae.config.scaling_factor
            mask_latent = F.interpolate(mask, size=masked_image_latent.shape[-2:], mode="nearest")
            densepose_latent = F.interpolate(densepose, size=masked_image_latent.shape[-2:], mode="nearest")

            # 2. prepare noise
            noise = torch.randn_like(masked_image_latent)
            self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.noise_scheduler.timesteps
            noise = noise * (getattr(self.noise_scheduler, 'init_noise_sigma', 1.0))
            latent = noise

            # 3. classifier-free guidance
            if do_classifier_free_guidance:
                # src_image_latent = torch.cat([src_image_latent] * 2)
                masked_image_latent = torch.cat([masked_image_latent] * 2)
                ref_image_latent = torch.cat([torch.zeros_like(ref_image_latent), ref_image_latent])
                mask_latent = torch.cat([mask_latent] * 2)
                densepose_latent = torch.cat([densepose_latent] * 2)

            # 4. Denoising loop
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            num_warmup_steps = (
                len(timesteps) - num_inference_steps * getattr(self.noise_scheduler, 'order', 1)
            )

            if ref_acceleration:
                down, reference_features = self.unet_encoder(
                    ref_image_latent, timesteps[num_inference_steps//2], encoder_hidden_states=None, return_dict=False
                )
                reference_features = list(reference_features)

            with tqdm.tqdm(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latent if we are doing classifier free guidance
                    _latent_model_input = (
                        torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                    )
                    _latent_model_input = self.noise_scheduler.scale_model_input(
                        _latent_model_input, t
                    )

                    # prepare the input for the inpainting model
                    latent_model_input = torch.cat(
                        [
                            _latent_model_input,
                            mask_latent,
                            masked_image_latent,
                            densepose_latent,
                        ],
                        dim=1,
                    )

                    if not ref_acceleration:
                        down, reference_features = self.unet_encoder(
                            ref_image_latent, t, encoder_hidden_states=None, return_dict=False
                        )
                        reference_features = list(reference_features)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=None,
                        reference_features=reference_features,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                    if do_classifier_free_guidance and guidance_scale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred,
                            noise_pred_cond,
                            guidance_rescale=guidance_scale,
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latent = self.noise_scheduler.step(
                        noise_pred, t, latent, **extra_step_kwargs, return_dict=False
                    )[0]
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.noise_scheduler.order == 0
                    ):
                        progress_bar.update()

            # Decode the final latent
            gen_image = latent_to_image(latent, self.vae)

            if repaint:
                src_image = (src_image / 2 + 0.5).clamp(0, 1)
                src_image = src_image.cpu().permute(0, 2, 3, 1).float().numpy()
                src_image = numpy_to_pil(src_image)
                mask = mask.cpu().permute(0, 2, 3, 1).float().numpy()
                mask = numpy_to_pil(mask)
                mask = [i.convert("RGB") for i in mask]
                gen_image = [
                    do_repaint(_src_image, _mask, _gen_image)
                    for _src_image, _mask, _gen_image in zip(src_image, mask, gen_image)
                ]

        return (gen_image,)


def latent_to_image(latent, vae):
    latent = 1 / vae.config.scaling_factor * latent
    image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L")
                      for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def do_repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 100
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled +
        (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg
