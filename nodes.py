import torch
import numpy as np
from PIL import Image
import os
import sys
import imageio
import logging
import json

import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.utils import ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

from diffusers import AutoencoderKLCosmos
from .CleanVAE import CleanVAE
from .pretrained_vae import VideoJITTokenizer, JointImageVideoTokenizer

from .diffusion_renderer_pipeline import CleanDiffusionRendererPipeline
from .model_diffusion_renderer import CleanDiffusionRendererModel
from .diffusion_renderer_config import (
    get_inverse_renderer_config,
    get_forward_renderer_config,
)

from .preprocess_envmap import (
    render_projection_from_panorama,
    tonemap_image_direct,
    latlong_vec,
    envmap_vec,
    clear_environment_cache,
    get_cache_stats,
    load_hdr_file,
    process_comfyui_tensor,
    rgb2srgb_official,
)

# Extracted utilities from the original codebase without dependencies
# Official mapping from cosmos_predict1/diffusion/inference/diffusion_renderer_utils/rendering_utils.py
GBUFFER_INDEX_MAPPING = {
    "basecolor": 0,
    "metallic": 1,
    "roughness": 2,
    "normal": 3,
    "depth": 4,
}

# Helper function to convert tensors to PIL images
def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

# Helper function to convert PIL images to tensors
def pil_to_tensor(image):
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)[None,]
    if tensor.ndim == 4 and tensor.shape[3] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor

def validate_frame_count(num_frames):
    """Ensure frame count is valid for Cosmos VAE (T=1 or T=1+8k).
    Returns the next valid count if invalid, padding up."""
    if num_frames == 1:
        return 1
    if (num_frames - 1) % 8 == 0:
        return num_frames
    next_valid = ((num_frames - 1) // 8 + 1) * 8 + 1
    return next_valid


class LoadDiffusionRendererModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "Models are loaded from 'ComfyUI/models/diffusion_models'"}),
            }
        }

    RETURN_TYPES = ("DIFFUSION_RENDERER_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Materia"

    def load_pipeline(self, model):
        device = mm.get_torch_device()
        dtype = torch.bfloat16
        print(f"Targeting device: {device}, dtype: {dtype}")

        # --- VAE LOADING (Simplified and Corrected) ---
        vae_main_dir = os.path.join(folder_paths.models_dir, "vae", "Cosmos-1.0-Tokenizer-CV8x8x8")
        if not vae_main_dir:
             raise FileNotFoundError("Could not find 'Cosmos-1.0-Tokenizer-CV8x8x8' in any VAE model directory.")
        
        image_vae_subfolder_path = os.path.join(vae_main_dir, "vae")
        if not os.path.isdir(image_vae_subfolder_path):
            raise FileNotFoundError(f"Image VAE subfolder not found at: {image_vae_subfolder_path}")

        # VAE stays on CPU until inference needs it.
        vae_instance = CleanVAE(model_path=image_vae_subfolder_path)
        vae_instance.reset_dtype(dtype)
        print("VAE loaded (CPU, will move to GPU during inference)")

        # --- MEMORY-EFFICIENT MODEL LOADING ---
        # Both models stay on CPU. The pipeline moves the active
        # model to GPU just-in-time for inference and back to CPU
        # afterward, so two 7B models can coexist on a 24GB card.
        checkpoint_path = folder_paths.get_full_path("diffusion_models", model)

        print(f"Loading checkpoint to CPU from: {checkpoint_path}")
        state_dict = comfy.utils.load_torch_file(
            checkpoint_path, safe_load=True
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]

        is_forward = "net.context_embedding.weight" not in state_dict
        if is_forward:
            print("Detected forward renderer checkpoint")
            basic_config = get_forward_renderer_config()
        else:
            print("Detected inverse renderer checkpoint")
            basic_config = get_inverse_renderer_config()

        print("Instantiating model skeleton on 'meta' device...")
        with torch.device("meta"):
            model_instance = CleanDiffusionRendererModel(basic_config)

        model_instance.to_empty(device='cpu')
        model_instance.to(dtype=dtype)
        model_instance.load_state_dict(state_dict, strict=True)
        del state_dict

        model_instance.net.reinit_non_persistent_buffers()
        model_instance.requires_grad_(False)
        model_instance.train(False)
        print(f"Model loaded (CPU, {dtype})")

        pipeline = CleanDiffusionRendererPipeline(
            checkpoint_dir=os.path.dirname(checkpoint_path),
            checkpoint_name=os.path.basename(checkpoint_path),
            model_type=None,
            vae_instance=vae_instance,
            model_instance=model_instance,
            guidance=2.0,
            num_steps=15,
            seed=42,
            dtype=dtype,
        )
        return (pipeline,)


class Cosmos1InverseRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("DIFFUSION_RENDERER_PIPELINE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "num_steps": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Denoising steps. More = better quality, slower."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("base_color", "metallic", "roughness", "normal", "depth")
    FUNCTION = "run_inverse_pass"
    CATEGORY = "Materia"

    def run_inverse_pass(self, pipeline, image, guidance=2.0, seed=42, num_steps=15):
        pipeline.set_model_type("inverse")
        pipeline.guidance = guidance
        pipeline.seed = seed
        pipeline.num_steps = num_steps

        # === ROBUST INPUT HANDLING ===
        pad_frames = 0
        if isinstance(image, list):
            try:
                image_5d = torch.stack(image, dim=0)
            except Exception as e:
                print(f"Warning: Could not stack list: {e}. "
                      f"Using first item only.")
                image_5d = image[0].unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image_5d = image.unsqueeze(0).unsqueeze(0)
            elif image.ndim == 4:
                B = image.shape[0]
                if B == 1:
                    image_5d = image.unsqueeze(1)  # (1,1,H,W,C)
                else:
                    image_5d = image.unsqueeze(0)  # (1,B,H,W,C) — B=T
                    T = image_5d.shape[1]
                    valid_T = validate_frame_count(T)
                    if valid_T != T:
                        pad_frames = valid_T - T
                        last_frame = image_5d[:, -1:].expand(
                            -1, pad_frames, -1, -1, -1
                        )
                        image_5d = torch.cat(
                            [image_5d, last_frame], dim=1
                        )
                        print(f"[Nodes] Padded {T} frames → {valid_T} "
                              f"(+{pad_frames} duplicated last frame)")
            elif image.ndim == 5:
                image_5d = image
            else:
                raise ValueError(
                    f"Unsupported tensor dimension: {image.ndim}")
        else:
            raise TypeError(
                f"Unsupported input type: {type(image)}")

        print(f"[Nodes] Input shape: {image_5d.shape} (B,T,H,W,C)")

        # === PRE-PROCESSING FOR MODEL ===
        image_tensor = image_5d.permute(0, 4, 1, 2, 3)
        image_tensor = image_tensor * 2.0 - 1.0
        print(f"[Nodes] Pre-processed input for model with shape: {image_tensor.shape}")
        
        # === INFERENCE ===
        inference_passes = ["basecolor", "metallic", "roughness", "normal", "depth"]
        outputs = {}
        pbar = ProgressBar(len(inference_passes))

        pipeline.to_gpu()
        for gbuffer_pass in inference_passes:
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            
            data_batch = {
                "rgb": image_tensor,
                "video": image_tensor,
                "context_index": torch.full((image_tensor.shape[0], 1), context_index, dtype=torch.long, device=image_tensor.device),
            }
            print(f"[Nodes] Running {gbuffer_pass} pass with context_index={context_index}")

            output_array = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(gbuffer_pass == 'normal'),
                seed=seed,
            )
            
            output_tensor = torch.from_numpy(output_array).float() / 255.0

            b, t, h, w, c = output_tensor.shape
            if pad_frames > 0:
                t = t - pad_frames
                output_tensor = output_tensor[:, :t]
            output_tensor_4d = output_tensor.reshape(b * t, h, w, c)

            outputs[gbuffer_pass] = output_tensor_4d
            pbar.update(1)

        pipeline.to_cpu()

        return (outputs["basecolor"], outputs["metallic"], outputs["roughness"], outputs["normal"], outputs["depth"])


class Cosmos1ForwardRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("DIFFUSION_RENDERER_PIPELINE",),
                "depth": ("IMAGE",),
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "base_color": ("IMAGE",),
                "env_map": ("IMAGE",),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "num_steps": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Denoising steps. More = better quality, slower."}),
                "env_format": (["proj", "ball", "direct"], {"default": "proj",
                    "tooltip": "proj: HDR cubemap projection. ball: direct tonemap. direct: passthrough (no tonemap/projection)."}),
                "env_brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "HDR intensity multiplier (applied before tonemapping)"}),
                "env_flip_horizontal": ("BOOLEAN", {"default": False}),
                "env_rotation": ("FLOAT", {"default": 180.0, "min": 0, "max": 360, "step": 1.0}),
                "max_point": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 100.0, "step": 0.5,
                    "tooltip": "Reinhard white point. Higher = more HDR highlight detail preserved."}),
                "log_scale": ("FLOAT", {"default": 10000.0, "min": 1.0, "max": 100000.0, "step": 100.0,
                    "tooltip": "Log encoding scale for env_log channel. Higher = more compression."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_forward_pass"
    CATEGORY = "Materia"

    def run_forward_pass(self, pipeline, depth, normal, roughness, metallic, base_color, env_map,
                        guidance=2.0, seed=42, num_steps=15, env_format="proj",
                        env_brightness=1.0, env_flip_horizontal=False,
                        env_rotation=180.0, max_point=16.0, log_scale=10000.0):

        pipeline.set_model_type("forward")
        pipeline.guidance = guidance
        pipeline.seed = seed
        pipeline.num_steps = num_steps

        gbuffer_tensors_in = {
            "depth": depth, "normal": normal, "roughness": roughness,
            "metallic": metallic, "base_color": base_color,
        }
        
        gbuffer_tensors_5d = {}
        pad_frames = 0
        for name, tensor_in in gbuffer_tensors_in.items():
            if isinstance(tensor_in, list):
                tensor_5d = torch.stack(tensor_in, dim=0)
            elif isinstance(tensor_in, torch.Tensor):
                if tensor_in.ndim == 3:
                    tensor_5d = tensor_in.unsqueeze(0).unsqueeze(0)
                elif tensor_in.ndim == 4:
                    B = tensor_in.shape[0]
                    if B == 1:
                        tensor_5d = tensor_in.unsqueeze(1)
                    else:
                        tensor_5d = tensor_in.unsqueeze(0)
                        T = tensor_5d.shape[1]
                        valid_T = validate_frame_count(T)
                        if valid_T != T:
                            pad_frames = valid_T - T
                            last = tensor_5d[:, -1:].expand(
                                -1, pad_frames, -1, -1, -1
                            )
                            tensor_5d = torch.cat(
                                [tensor_5d, last], dim=1
                            )
                elif tensor_in.ndim == 5:
                    tensor_5d = tensor_in
                else:
                    raise ValueError(
                        f"Unsupported dim for '{name}': "
                        f"{tensor_in.ndim}")
            else:
                raise TypeError(
                    f"Unsupported type for '{name}': "
                    f"{type(tensor_in)}")
            gbuffer_tensors_5d[name] = tensor_5d
        
        B, T, H, W, C = gbuffer_tensors_5d["depth"].shape
        device = mm.get_torch_device()
        
        data_batch = {}
        key_mapping = {"base_color": "basecolor", "depth": "depth", "normal": "normal", 
                       "roughness": "roughness", "metallic": "metallic"}

        for name, tensor_5d in gbuffer_tensors_5d.items():
            processed_tensor = tensor_5d.permute(0, 4, 1, 2, 3) * 2.0 - 1.0
            data_batch[key_mapping[name]] = processed_tensor
        
        data_batch['video'] = data_batch['depth']

        envlight_dict = None
        if env_format == 'proj':
            print("[Nodes] Processing env_map as panoramic projection ('proj' mode).")
            envlight_dict = render_projection_from_panorama(
                env_input=env_map, resolution=(H, W), num_frames=T,
                device=device, env_brightness=env_brightness,
                env_flip=env_flip_horizontal, env_rot=env_rotation,
                max_point=max_point, log_scale=log_scale,
            )
        elif env_format == 'ball':
            print("[Nodes] Processing env_map as direct tonemap ('ball' mode).")
            if H != W:
                logging.warning(
                    f"Ball mode expects square input, "
                    f"but G-buffers are {W}x{H}."
                )
            envlight_dict = tonemap_image_direct(
                env_input=env_map, resolution=(H, W), num_frames=T,
                device=device, max_point=max_point, log_scale=log_scale,
                env_brightness=env_brightness,
                env_flip=env_flip_horizontal, env_rot=env_rotation,
            )
        elif env_format == 'direct':
            print("[Nodes] Processing env_map in 'direct' mode "
                  "(passthrough, no tonemap/projection).")
            env_tensor = process_comfyui_tensor(env_map).to(device).float()
            if env_brightness != 1.0:
                env_tensor = env_tensor * env_brightness
            if env_flip_horizontal:
                env_tensor = torch.flip(env_tensor, dims=[1])
            if env_rotation != 0:
                env_w = env_tensor.shape[1]
                pixel_rot = int(env_w * env_rotation / 360)
                env_tensor = torch.roll(
                    env_tensor, shifts=pixel_rot, dims=1,
                )
            if env_tensor.shape[0] != H or env_tensor.shape[1] != W:
                env_tensor = torch.nn.functional.interpolate(
                    env_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W), mode='bilinear', align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            env_ldr_frame = env_tensor.clamp(0, 1)
            env_log_frame = rgb2srgb_official(
                torch.log1p(env_tensor.clamp(min=0))
                / np.log1p(log_scale)
            ).clamp(0, 1)
            if T > 1:
                env_ldr_out = env_ldr_frame.unsqueeze(0).expand(
                    T, -1, -1, -1)
                env_log_out = env_log_frame.unsqueeze(0).expand(
                    T, -1, -1, -1)
            else:
                env_ldr_out = env_ldr_frame.unsqueeze(0)
                env_log_out = env_log_frame.unsqueeze(0)
            envlight_dict = {
                'env_ldr': env_ldr_out, 'env_log': env_log_out,
            }

        env_ldr = (envlight_dict['env_ldr']
                   .permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
        env_log = (envlight_dict['env_log']
                   .permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0)
        env_nrm_01 = envmap_vec(res=(H, W), device=device) * 0.5 + 0.5
        env_nrm = (env_nrm_01.permute(2, 0, 1)
                   .unsqueeze(0).unsqueeze(2) * 2.0 - 1.0)

        data_batch['env_ldr'] = env_ldr.expand(B, -1, -1, -1, -1)
        data_batch['env_log'] = env_log.expand(B, -1, -1, -1, -1)
        data_batch['env_nrm'] = env_nrm.expand(B, -1, T, -1, -1)

        print(f"[Nodes] env_format={env_format}, max_point={max_point}, "
              f"log_scale={log_scale}, brightness={env_brightness}")
        print(f"[Nodes] env_ldr range: "
              f"[{envlight_dict['env_ldr'].min():.3f}, "
              f"{envlight_dict['env_ldr'].max():.3f}]")
        print(f"[Nodes] env_log range: "
              f"[{envlight_dict['env_log'].min():.3f}, "
              f"{envlight_dict['env_log'].max():.3f}]")
        print(f"[Nodes] env_nrm range (post-norm): "
              f"[{env_nrm.min():.3f}, {env_nrm.max():.3f}]")
        env_hash = hash(
            (data_batch['env_ldr'].sum().item(),
             data_batch['env_log'].sum().item())
        )
        print(f"[Nodes] env condition hash: {env_hash:#018x} "
              "(should change when env params change)")
        print("[Nodes] Data batch prepared. Calling diffusion pipeline...")
        pipeline.to_gpu()
        output_array = pipeline.generate_video(data_batch=data_batch, seed=seed)
        pipeline.to_cpu()

        output_tensor = torch.from_numpy(output_array).float() / 255.0

        b, t, h, w, c = output_tensor.shape
        if pad_frames > 0:
            t = t - pad_frames
            output_tensor = output_tensor[:, :t]
        final_output = output_tensor.reshape(b * t, h, w, c)

        return (final_output,)

class LoadHDRImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"tooltip": "Path to HDR image (.hdr, .exr)"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_hdr"
    CATEGORY = "Materia"

    def load_hdr(self, path):
        tensor = load_hdr_file(path).unsqueeze(0)
        return (tensor,)
    
class VAEPassthroughTest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae_path": ("STRING", {"default": "Cosmos-1.0-Tokenizer-CV8x8x8/vae"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("output", "diff_image", "stats")
    FUNCTION = "test_vae"
    CATEGORY = "Materia/Debug"

    def test_vae(self, image, vae_path):
        import torch
        import numpy as np
        import os
        import folder_paths
        from .CleanVAE import CleanVAE
        
        # Initialize VAE
        vae_full_path = os.path.join(folder_paths.models_dir, "vae", vae_path)
        vae = CleanVAE(model_path=vae_full_path)
        vae.to(torch.device('cuda'))
        
        # Use float32 for best quality in testing
        vae.reset_dtype(torch.float32)
        
        # Prepare input - convert to 5D tensor in [-1, 1]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        
        # Handle dimensions
        if image.ndim == 3:  # (H, W, C)
            image = image.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W, C)
        elif image.ndim == 4:  # (B, H, W, C)
            image = image.unsqueeze(1)  # -> (B, 1, H, W, C)
        elif image.ndim == 5:  # Already (B, T, H, W, C)
            pass
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
        
        B, T, H, W, C = image.shape
        
        # Convert to model format: (B, C, T, H, W) in range [-1, 1]
        image_tensor = image.permute(0, 4, 1, 2, 3)
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Move to GPU with float32 for best precision
        image_tensor = image_tensor.to(device='cuda', dtype=torch.float32)
        
        print(f"\n{'='*60}")
        print(f"VAE PASSTHROUGH TEST (float32 precision)")
        print(f"{'='*60}")
        print(f"Input shape: {image_tensor.shape}")
        print(f"Input range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
        print(f"Input mean: {image_tensor.mean():.3f}, std: {image_tensor.std():.3f}")
        
        # Encode
        with torch.no_grad():
            latent = vae.encode(image_tensor)
            print(f"\nLatent shape: {latent.shape}")
            print(f"Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"Latent mean: {latent.mean():.3f}, std: {latent.std():.3f}")
            
            # Decode
            reconstructed = vae.decode(latent)
            print(f"\nReconstructed shape: {reconstructed.shape}")
            print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            print(f"Reconstructed mean: {reconstructed.mean():.3f}, std: {reconstructed.std():.3f}")
        
        # Calculate difference
        diff = (reconstructed - image_tensor).abs()
        mse = ((reconstructed - image_tensor) ** 2).mean()
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # PSNR for [-1,1] range
        
        print(f"\nReconstruction Metrics:")
        print(f"  L1 Error: {diff.mean():.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Max Error: {diff.max():.6f}")
        
        # Try different output normalizations
        print(f"\n{'='*40}")
        print("Testing different output normalizations:")
        print(f"{'='*40}")
        
        # Method 1: Standard [-1,1] to [0,1]
        output1 = (reconstructed + 1.0) / 2.0
        print(f"Method 1 (standard): [{output1.min():.3f}, {output1.max():.3f}]")
        
        # Method 2: Clamp first then normalize (for outputs slightly outside [-1,1])
        output2 = (reconstructed.clamp(-1, 1) + 1.0) / 2.0
        print(f"Method 2 (clamp then normalize): [{output2.min():.3f}, {output2.max():.3f}]")
        
        # Method 3: Min-max normalization
        if reconstructed.max() > reconstructed.min():
            output3 = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
            print(f"Method 3 (min-max): [{output3.min():.3f}, {output3.max():.3f}]")
        else:
            output3 = output2  # Fallback if constant
        
        # Use method 2 for output (clamp then normalize - safest)
        output_tensor = output2.clamp(0, 1)
        
        # Convert back to ComfyUI format (B, T, H, W, C)
        output_tensor = output_tensor.permute(0, 2, 3, 4, 1)
        output_tensor = output_tensor.reshape(B * T, H, W, C)
        
        # Create difference visualization (amplify for visibility)
        diff_vis = diff * 5.0  # Amplify difference for visualization
        diff_vis = diff_vis.clamp(0, 1)
        diff_tensor = diff_vis.permute(0, 2, 3, 4, 1)
        diff_tensor = diff_tensor.reshape(B * T, H, W, C)
        
        # Create stats string
        stats = f"""VAE Test Results (float32):
Input: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]
Latent: [{latent.min():.3f}, {latent.max():.3f}]
Output: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]

Quality Metrics:
  L1 Error: {diff.mean():.6f}
  MSE: {mse:.6f}
  PSNR: {psnr:.2f} dB
  Max Error: {diff.max():.6f}

Statistics:
  Output Mean: {reconstructed.mean():.3f} (ideal: ~0)
  Output Std: {reconstructed.std():.3f} (ideal: ~0.5)
"""
        
        print(f"\n{stats}")
        print(f"{'='*60}\n")
        
        return (output_tensor.cpu(), diff_tensor.cpu(), stats)


NODE_CLASS_MAPPINGS = {
    "MateriaLoadModel": LoadDiffusionRendererModel,
    "MateriaInverseRenderer": Cosmos1InverseRenderer,
    "MateriaForwardRenderer": Cosmos1ForwardRenderer,
    "MateriaLoadHDR": LoadHDRImage,
    "MateriaVAETest": VAEPassthroughTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MateriaLoadModel": "Load Materia Model",
    "MateriaInverseRenderer": "Inverse Rendering (Image to PBR)",
    "MateriaForwardRenderer": "Forward Rendering (PBR to Image)",
    "MateriaLoadHDR": "Load HDR Image",
    "MateriaVAETest": "VAE Passthrough Test",
}