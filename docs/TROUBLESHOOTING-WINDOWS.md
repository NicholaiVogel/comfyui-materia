# Windows Troubleshooting Guide for Materia

This guide helps resolve common Windows installation issues for ComfyUI Materia custom nodes.

---

## Table of Contents

- [Common Installation Errors](#common-installation-errors)
- [Prerequisites](#prerequisites)
- [Visual Studio Build Tools Installation](#visual-studio-build-tools-installation)
- [CUDA Toolkit Installation](#cuda-toolkit-installation)
- [Specific Error Solutions](#specific-error-solutions)
- [Alternative Installation Methods](#alternative-installation-methods)
- [Verifying Your Installation](#verifying-your-installation)
- [Getting Help](#getting-help)

---

## Common Installation Errors

These are the most common error messages users encounter on Windows:

### Error 1: "Failed to build nvdiffrast"

```
error: Microsoft Visual C++ 14.0 or greater is required.
Get it with "Microsoft C++ Build Tools"
```

**Cause:** Missing or incomplete Visual Studio Build Tools.

**Solution:** See [Visual Studio Build Tools Installation](#visual-studio-build-tools-installation) below.

---

### Error 2: "Could not find Visual Studio compiler"

```
RuntimeError: Ninja is required to load C++ extensions
```

**Cause:** C++ compiler not found in PATH or build tools not properly installed.

**Solution:** 
1. Ensure Visual Studio Build Tools are installed (see below)
2. Run Developer Command Prompt for VS, then retry installation

---

### Error 3: "diffusers version too old"

```
ImportError: cannot import name 'AutoencoderKLCosmos' from 'diffusers'
```

**Cause:** `diffusers` version is less than 0.33.0.

**Solution:**
```cmd
cd ComfyUI\custom_nodes\comfyui-materia
python install.py
```
The installer will automatically upgrade diffusers to the required version.

---

### Error 4: "CUDA not available"

```
RuntimeError: CUDA not available
```

**Cause:** PyTorch was installed without CUDA support or CUDA drivers are outdated.

**Solution:** See [CUDA Toolkit Installation](#cuda-toolkit-installation) below.

---

## Prerequisites

Before installing Materia on Windows, ensure you have:

- **Windows 10 or 11** (64-bit)
- **NVIDIA GPU** with 24GB+ VRAM recommended
- **Python 3.10 or 3.11** (installed with ComfyUI)
- **CUDA Toolkit 11.8 or 12.x** matching your PyTorch version
- **Visual Studio Build Tools 2019 or 2022**

---

## Visual Studio Build Tools Installation

nvdiffrast requires C++ build tools to compile from source on Windows.

### Step 1: Download Visual Studio Installer

1. Go to [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/)
2. Download **Visual Studio Community 2022** (Free)

### Step 2: Install Build Tools

1. Run the installer
2. Select **"Desktop development with C++"**
3. On the right sidebar, ensure these are checked:
   - ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
   - ✅ Windows 10 SDK (or Windows 11 SDK)
   - ✅ C++ CMake tools for Windows

4. Click **Install**

### Step 3: Restart Your Computer

**IMPORTANT:** After installation completes, you MUST restart Windows for the build tools to be properly registered.

### Step 4: Verify Installation

Open **Command Prompt** and run:
```cmd
where cl.exe
```

If it returns a path like:
```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.xx.xxxx\bin\Hostx64\x64\cl.exe
```

Your installation is correct!

### Alternative: Lightweight Installation

If you don't want to install full Visual Studio:

1. Download **"Build Tools for Visual Studio 2022"** from the same page
2. During installation, select only:
   - C++ build tools
   - MSVC v143 build tools
   - Windows 10/11 SDK

This uses less disk space (~6GB vs 20GB).

---

## CUDA Toolkit Installation

### Check Your CUDA Version

1. Open **Command Prompt**
2. Run:
```cmd
nvcc --version
```

If you see version info, CUDA is installed. Note the version (e.g., 12.1).

### Check PyTorch CUDA Version

1. Open **Python** (from ComfyUI directory):
```cmd
cd ComfyUI
python_embeded\python.exe
```

2. Run:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
```

### Install Matching CUDA Toolkit

If CUDA is missing or version mismatched:

1. Go to [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Download the version matching your PyTorch CUDA version
3. Run the installer
4. Accept all defaults
5. Restart your computer

### Common CUDA Versions

| PyTorch CUDA | Recommended CUDA Toolkit | Download Link |
|--------------|------------------------|--------------|
| cu121 | CUDA 12.1 | [Download](https://developer.nvidia.com/cuda-12-1-0-download-archive) |
| cu124 | CUDA 12.4 | [Download](https://developer.nvidia.com/cuda-12-4-0-download-archive) |
| cu118 | CUDA 11.8 | [Download](https://developer.nvidia.com/cuda-11-8-0-download-archive) |

---

## Specific Error Solutions

### "Failed to build nvdiffrast" - Full Fix

If you see this error despite having Visual Studio installed:

1. **Check for conflicting installations:**
   ```cmd
   where cl.exe
   ```
   If multiple paths appear, you may have conflicting installations.

2. **Use Developer Command Prompt:**
   - Press Windows key
   - Search for "Developer Command Prompt for VS 2022"
   - Right-click → Run as administrator
   - Navigate to ComfyUI:
     ```cmd
     cd C:\Path\To\ComfyUI\custom_nodes\comfyui-materia
     python install.py
     ```

3. **Install build dependencies manually:**
   ```cmd
   python -m pip install setuptools wheel ninja
   ```

4. **Try installing nvdiffrast manually:**
   ```cmd
   python -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
   ```

5. **If still failing:** Try the pre-built diffrp-nvdiffrast package instead:
   ```cmd
   python -m pip install diffrp-nvdiffrast
   ```

### "Microsoft Visual C++ 14.0 is required"

This error means C++ build tools are missing or too old.

**Fix:** Follow the [Visual Studio Build Tools Installation](#visual-studio-build-tools-installation) section above.

### "Permission denied" or "Access denied"

**Fix:** Run Command Prompt as Administrator:
1. Press Windows key
2. Type "cmd"
3. Right-click "Command Prompt" → Run as administrator

### "pip is not recognized"

**Fix:** You're not using ComfyUI's Python. Always use:
```cmd
cd ComfyUI
python_embeded\python.exe -m pip install ...
```

Or just run:
```cmd
cd ComfyUI\custom_nodes\comfyui-materia
python install.py
```

---

## Alternative Installation Methods

### Method 1: Using Conda (Recommended for Advanced Users)

If you have issues with the portable Python environment:

1. **Install Miniconda:**
   - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)
   - Install with default settings

2. **Create a dedicated environment:**
   ```cmd
   conda create -n materia python=3.11 -y
   conda activate materia
   ```

3. **Install PyTorch with CUDA:**
   ```cmd
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. **Install Materia dependencies:**
   ```cmd
   cd ComfyUI\custom_nodes\comfyui-materia
   pip install -r requirements.txt
   pip install diffrp-nvdiffrast
   ```

5. **Point ComfyUI to this environment:**
   - Set environment variable `PYTHON` to your conda Python path
   - Or configure ComfyUI to use this Python

### Method 2: Using Virtual Environment

```cmd
cd ComfyUI
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
cd custom_nodes\comfyui-materia
python install.py
```

### Method 3: Manual Installation (Last Resort)

If all else fails, install dependencies manually:

```cmd
cd ComfyUI\custom_nodes\comfyui-materia

python -m pip install numpy>=1.23.0
python -m pip install Pillow>=9.0.0
python -m pip install imageio>=2.22.0
python -m pip install opencv-python>=4.7.0
python -m pip install einops>=0.6.0
python -m pip install safetensors>=0.3.0
python -m pip install typing-extensions>=4.0.0
python -m pip install "diffusers>=0.33.0"

# Try nvdiffrast (may fail without build tools)
python -m pip install diffrp-nvdiffrast
```

---

## Verifying Your Installation

After installation, verify everything is working:

### Test 1: Check Package Versions

Open Python in ComfyUI directory:
```cmd
cd ComfyUI
python_embeded\python.exe
```

Run:
```python
import diffusers
print(f"diffusers: {diffusers.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import nvdiffrast.torch as dr
    print("nvdiffrast: ✓")
except ImportError:
    print("nvdiffrast: ✗ (forward renderer won't work)")

try:
    from diffusers import AutoencoderKLCosmos
    print("AutoencoderKLCosmos: ✓")
except ImportError:
    print("AutoencoderKLCosmos: ✗")
```

**Expected output:**
```
diffusers: 0.34.0  (or higher)
PyTorch: 2.x.x+cu121
CUDA available: True
nvdiffrast: ✓
AutoencoderKLCosmos: ✓
```

### Test 2: Start ComfyUI

1. Start ComfyUI normally
2. Look for Materia nodes in the node menu under "Materia"
3. If nodes appear but fail to load, check ComfyUI console for errors

### Test 3: Test Forward Renderer

If nvdiffrast is installed, try loading a forward renderer model:

1. Load "Load Materia Model" node
2. Select a forward renderer checkpoint
3. If it loads without error, installation is successful!

---

## Getting Help

If you're still having issues after following this guide:

### 1. Check the GitHub Issues

- Search existing issues: [comfyui-materia issues](https://github.com/NicholaiVogel/comfyui-materia/issues)
- Your problem may already be solved

### 2. Collect Debug Information

Before asking for help, gather this information:

```cmd
# System info
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"

# Python info
python --version

# PyTorch info
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Package versions
python -m pip list | findstr "diffusers nvdiffrast torch"

# CUDA info
nvcc --version
```

### 3. Create a Detailed Bug Report

When opening an issue, include:

1. **Error message** (full output, not just the error line)
2. **Steps to reproduce** (what you did)
3. **System information** (from above)
4. **Installation method** (portable, venv, conda, etc.)
5. **Screenshots** of error messages (if applicable)

### 4. Quick Reference: Common Solutions Summary

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| "Failed to build nvdiffrast" | Missing VS Build Tools | Install Visual Studio Build Tools |
| "diffusers version too old" | Old diffusers version | Run `python install.py` to auto-upgrade |
| "CUDA not available" | Wrong PyTorch version | Install PyTorch with CUDA support |
| "cl.exe not found" | Build tools not in PATH | Use Developer Command Prompt |
| Import errors | Wrong Python environment | Use ComfyUI's python_embeded |

---

## Tips for Successful Installation

1. **Always use ComfyUI's Python:** Never install into system Python if using ComfyUI Portable
2. **Restart after installing build tools:** Visual Studio requires restart to register compilers
3. **Match CUDA versions:** PyTorch CUDA version must match installed CUDA Toolkit
4. **Run as admin:** If you get permission errors, run Command Prompt as Administrator
5. **Use the install script:** `python install.py` handles most issues automatically
6. **Check disk space:** Build tools require ~6GB, CUDA Toolkit ~3GB
7. **Disable antivirus temporarily:** Some AV software blocks build processes

---

## Additional Resources

- [nvdiffrast GitHub](https://github.com/NVlabs/nvdiffrast)
- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [ComfyUI Installation](https://github.com/comfyanonymous/ComfyUI)
- [Visual Studio C++ Workloads](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation)

---

## FAQ

### Q: Do I really need 24GB VRAM?

**A:** For both inverse and forward renderers simultaneously, yes. You can use just the inverse renderer (image → PBR) with ~12GB VRAM.

### Q: Can I skip nvdiffrast installation?

**A:** Yes, but the forward renderer (PBR → relit image) won't work. The inverse renderer will still function.

### Q: Is there a pre-built nvdiffrast package?

**A:** Yes! The install script tries `diffrp-nvdiffrast` first, which is pre-built and easier to install.

### Q: Why does Windows make this so hard?

**A:** Windows doesn't have system-level package management like Linux's apt/pacman, so build tools must be installed separately.

### Q: Can I use WSL (Windows Subsystem for Linux)?

**A:** Yes! WSL2 with Ubuntu is often easier for Linux-based AI tools. However, GPU support in WSL2 can be tricky.

---

Last updated: January 2026
