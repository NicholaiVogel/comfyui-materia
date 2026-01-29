Materia
===

*by [Biohazard VFX](https://biohazardvfx.com)*

Materia is a set of custom nodes by Biohazard VFX for ComfyUI. 

The first set of nodes in this toolkit are our implementation 
of Nvidia's [cosmos-transfer1-diffusionrenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) video models.

- **Inverse Rendering (Image to PBR)** - Decomposes a single
RGB video or image into PBR material maps (base color, metallic,
roughness, normal, depth)    
- **Forward Rendering (PBR to Image)** - Applies the lighting
from an input HDRI (.hdr,.exr) to relight the input RGB 
video/image. 

Example workflow
---

> **Notice** For a good introduction, we recommend running the
example workflow included in this repository, which contains all
of the elements required for inference + our custom nodes 
preconfigured in a stable/tested pipeline.

The workflow has been optimized for nvidia GPU's equipped with 
24gb or more of vram + 64gb of system memory, but theoretically 
16gb of vram should be possible. We will continue to make 
changes to the example workflow and the inference pipeline as 
time goes on to ensure accessibilty/reliability for the users of 
these tools.

Nodes
---

All nodes appear under the **Materia** category in ComfyUI.

| Node | What it does |
|------|-------------|
| **Load Materia Model** | Loads a diffusion renderer checkpoint + Cosmos VAE |
| **Inverse Rendering (Image to PBR)** | RGB image -> 5 G-buffer maps |
| **Forward Rendering (PBR to Image)** | G-buffers + environment map -> relit RGB |
| **Load HDR Image** | Loads .hdr / .exr environment maps |
| **VAE Passthrough Test** | Debug node for testing VAE encode/decode quality |

Installation
---

> Prerequisites:
> [Python](https://www.python.org/)
> [Git](https://git-scm.com/install/)
> Windows only: [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/)


**Clone into custom_nodes:**

Make sure you are in your custom nodes directory:

```bash
cd ~/ComfyUI/custom_nodes
```

Then run this: (requires git)

```
git clone https://github.com/NicholaiVogel/comfyui-materia.git
```

**Install dependencies:**

Go into the `comfyui-materia` directory

```bash
cd comfyui-materia
```
Run the automated installer:

```bash
python install.py
```

The installer will:
- Detect your ComfyUI Python environment (portable, venv, Aki, system)
- Install/upgrade diffusers to >= 0.33.0
- Install nvdiffrast (with automatic fallbacks)
- Verify installation

**Windows users:** If the installer fails, see [TROUBLESHOOTING-WINDOWS.md](docs/TROUBLESHOOTING-WINDOWS.md) for detailed troubleshooting steps.

**Linux/Mac users:** If nvdiffrast installation fails, install system build tools first:

```bash
# debian/ubuntu
sudo apt-get install -y build-essential cmake pkg-config \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev

# arch (btw)
sudo pacman -S base-devel cmake pkgconf libglvnd mesa
```

Then re-run `python install.py`.

### Models

**1. Diffusion Renderer checkpoints**

Download from
[huggingface.co/collections/zianw/cosmos-diffusionrenderer](https://huggingface.co/collections/zianw/cosmos-diffusionrenderer-6849f2a4da267e55409b8125)
and place in `ComfyUI/models/diffusion_models/`:

```
models/diffusion_models/
├── Diffusion_Renderer_Forward_Cosmos_7B/
│   ├── config.json
│   └── model.pt
├── Diffusion_Renderer_Inverse_Cosmos_7B/
│   ├── config.json
│   └── model.pt
```

**2. Cosmos Video Tokenizer (VAE)**

```bash
cd ~/ComfyUI/models/vae
huggingface-cli download nvidia/Cosmos-1.0-Tokenizer-CV8x8x8 \
    --local-dir Cosmos-1.0-Tokenizer-CV8x8x8
```

Keep the `vae/` subfolder structure. JIT tokenizers can be deleted. 

```
models/vae/
├── Cosmos-1.0-Tokenizer-CV8x8x8/
│   └── vae/
│       ├── config.json <----- note: this NEEDS to be here for the VAE to be loaded. 
│       └── diffusion_pytorch_model.safetensors
```

Node Reference
---

### Load Materia Model

Loads a diffusion renderer pipeline (model + Cosmos VAE). The node
auto-detects whether the checkpoint is an inverse or forward renderer based
on the state dict.

Both models can coexist on a 24GB GPU -- the pipeline manages CPU/GPU
offloading automatically.

| Input | Type | Description |
|-------|------|-------------|
| model | selection | Checkpoint from `diffusion_models/` |

| Output | Type |
|--------|------|
| DIFFUSION_RENDERER_PIPELINE | pipeline object |


### Inverse Rendering (Image to PBR)

Decomposes a single RGB image into 5 PBR G-buffer maps. Runs 5 sequential
diffusion passes, one per buffer, using a context embedding to select which
material property to predict.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | PIPELINE | -- | from Load Materia Model (inverse) |
| image | IMAGE | -- | input RGB image |
| guidance | FLOAT | 2.0 | classifier-free guidance scale (0-10) |
| seed | INT | 42 | random seed |
| num_steps | INT | 15 | denoising steps (1-100) |

| Output | Type | Description |
|--------|------|-------------|
| base_color | IMAGE | albedo / diffuse color |
| metallic | IMAGE | metallic map |
| roughness | IMAGE | roughness map |
| normal | IMAGE | surface normal map |
| depth | IMAGE | depth map |


### Forward Rendering (PBR to Image)

Relights a subject from its G-buffers under a new environment map. Runs a
single diffusion pass conditioned on all 5 G-buffers + environment lighting.
Requires nvdiffrast for the `proj` environment format.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | PIPELINE | -- | from Load Materia Model (forward) |
| depth | IMAGE | -- | depth map |
| normal | IMAGE | -- | normal map |
| roughness | IMAGE | -- | roughness map |
| metallic | IMAGE | -- | metallic map |
| base_color | IMAGE | -- | albedo map |
| env_map | IMAGE | -- | environment lighting |
| guidance | FLOAT | 2.0 | classifier-free guidance (0-10) |
| seed | INT | 42 | random seed |
| num_steps | INT | 15 | denoising steps (1-100) |
| env_format | ENUM | "proj" | environment map format (see below) |
| env_brightness | FLOAT | 1.0 | HDR intensity multiplier (0-10) |
| env_flip_horizontal | BOOL | false | flip environment horizontally |
| env_rotation | FLOAT | 180.0 | rotation in degrees (0-360) |
| max_point | FLOAT | 16.0 | Reinhard tone map white point (0.1-100) |
| log_scale | FLOAT | 10000.0 | log encoding scale (1-100000) |

| Output | Type |
|--------|------|
| IMAGE | relit RGB image |

**Environment map formats:**

- **proj** -- equirectangular panorama projected to a cubemap perspective
  view via nvdiffrast. this is the standard mode for HDR environment maps.
- **ball** -- direct tonemapping, treats the input as a chrome ball capture.
  no cubemap projection.
- **direct** -- passthrough with optional brightness/flip/rotation. no
  tonemapping or projection applied. use this for pre-processed lighting.


### Load HDR Image

Loads high dynamic range images from disk. Supports `.hdr` and `.exr`
formats via imageio and OpenCV.

| Input | Type | Description |
|-------|------|-------------|
| path | STRING | absolute file path to the HDR image |

| Output | Type |
|--------|------|
| IMAGE | HDR tensor |


Pipeline Workflow
---

An example workflow is included at `examples/materia_pipeline.json`. Load it
in ComfyUI via **Load Workflow**.

The pipeline flow:

```
Load Image
    │
    ├──> Load Materia Model (Inverse)
    │        │
    │        └──> Inverse Rendering ──> base_color, metallic,
    │                                    roughness, normal, depth
    │
    ├──> Load Materia Model (Forward)
    │        │
    │        └──> Forward Rendering <── G-buffers + env_map ──> relit RGB
    │
    ├──> Depth Anything V2 ──> depth preview
    │
    ├──> InSPyReNet ──> alpha matte
    │
    └──> DepthToNormalMap ──> normal preview
```

The workflow also includes an optional post-processing subgraph that
upscales the G-buffer outputs back to input resolution via lanczos and
applies detail transfer for sharpening.


Recommended Settings
---

- **Resolution**: the model operates at 1024x1024 internally. for best
  results, scale your input to 1024 before the inverse pass.
- **Guidance**: 1.0 - 2.0. higher values increase contrast but may
  introduce artifacts.
- **Steps**: 8-15. diminishing returns above 15.
- **VRAM**: 24GB minimum. both 7B models share the GPU via automatic
  CPU offloading -- only one is on GPU at a time.


Architecture
---

The backbone is NVIDIA's FADIT (Frame-Aware Diffusion Transformer): 4096
hidden dim, 28 blocks, 32 heads, ~7B parameters. Uses EDM preconditioning
(sigma_data=0.5) with the diffusers `EDMEulerScheduler`.

### Inverse rendering

The input RGB is encoded to latent space via the Cosmos 8x8x8 VAE,
concatenated with the noisy latent, and a context embedding selects which
G-buffer to predict (0=basecolor, 1=metallic, 2=roughness, 3=normal,
4=depth). Each buffer requires a separate diffusion pass.

### Forward rendering

All 5 G-buffers are encoded to latent space and concatenated with the noisy
latent. Environment lighting is represented as 3 channels: tone-mapped LDR,
log-encoded HDR, and direction normal vectors. The full conditioning is 136
channels (8 conditions x 17 channels each).

### Key implementation details

- Memory-efficient loading via meta device + bfloat16 (fits 24GB VRAM)
- Positional embedding buffers reinitialized after meta-device
  materialization (non-persistent buffers aren't restored by
  load_state_dict)
- VAE latents normalized with per-channel mean/std from Cosmos config
- sigma_min=0.02 (DiffusionRenderer-specific, not the base Cosmos value
  of 0.0002)
- VAE uses float32 internally for encode/decode quality regardless of
  model dtype
- Video support: valid frame counts are T=1 or T=1+8k (1, 9, 17, 25...)
- HDR environment maps are cached (LRU, 10 entries) keyed on resolution +
  brightness + rotation parameters
- Normal maps receive special post-processing: normalization + blend ratio
  based on magnitude to handle low-confidence regions


Contributing
---

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs and
submitting pull requests.


License
---

The ComfyUI-materia nodes are licensed under the GNU General Public License v3.0

Original model weights for [cosmos-transfer1-diffusionrenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) by NVIDIA 
under an Apache-2.0 license. 
