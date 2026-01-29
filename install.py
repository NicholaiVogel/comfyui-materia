import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional

try:
    from pkg_resources import get_distribution as get_dist
except ImportError:
    from importlib.metadata import version as get_dist


class ColoredOutput:
    """ANSI color codes for terminal output"""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    @staticmethod
    def success(msg: str):
        print(f"{ColoredOutput.GREEN}✓ {msg}{ColoredOutput.RESET}")

    @staticmethod
    def warning(msg: str):
        print(f"{ColoredOutput.YELLOW}⚠ {msg}{ColoredOutput.RESET}")

    @staticmethod
    def error(msg: str):
        print(f"{ColoredOutput.RED}✗ {msg}{ColoredOutput.RESET}")

    @staticmethod
    def info(msg: str):
        print(f"{ColoredOutput.BLUE}ℹ {msg}{ColoredOutput.RESET}")


def get_python_executable() -> Tuple[str, str]:
    """Detect the correct Python executable for ComfyUI installation"""
    script_dir = Path(__file__).parent

    candidates = [
        (
            "ComfyUI Portable",
            script_dir / "../../../python_embeded/python.exe"
            if os.name == "nt"
            else script_dir / "../../../python_embeded/python",
        ),
        (
            "ComfyUI Aki",
            script_dir / "../../python/python.exe"
            if os.name == "nt"
            else script_dir / "../../python/python",
        ),
        ("System Python", Path(sys.executable)),
    ]

    for name, path in candidates:
        if path.exists():
            ColoredOutput.success(f"Found {name}: {path}")
            return str(path), name

    ColoredOutput.warning(
        f"No ComfyUI Python found, using system Python: {sys.executable}"
    )
    return sys.executable, "System Python"


def run_pip(python_exec: str, *args, capture: bool = False) -> Tuple[bool, str]:
    """Run pip command and return success status and output"""
    try:
        if capture:
            result = subprocess.run(
                [python_exec, "-m", "pip", "install", "--no-warn-script-location"]
                + list(args),
                capture_output=True,
                text=True,
                check=True,
            )
            return True, result.stdout
        else:
            subprocess.run(
                [python_exec, "-m", "pip", "install", "--no-warn-script-location"]
                + list(args),
                check=True,
            )
            return True, ""
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return False, error_msg


def get_package_version(package: str) -> Optional[str]:
    """Get installed package version"""
    try:
        version = get_dist(package)
        # pkg_resources returns Distribution object with .version attribute
        # importlib.metadata.version returns string directly
        if hasattr(version, "version"):
            return version.version
        else:
            return str(version)
    except Exception:
        return None


def check_torch_cuda(python_exec: str) -> bool:
    """Check if PyTorch with CUDA is available"""
    try:
        code = """
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
else:
    print("CUDA not available")
    exit(1)
"""
        result = subprocess.run(
            [python_exec, "-c", code], capture_output=True, text=True
        )

        if result.returncode == 0:
            ColoredOutput.success(result.stdout.strip())
            return True
        else:
            ColoredOutput.warning("PyTorch CUDA not available")
            return False
    except Exception as e:
        ColoredOutput.warning(f"Could not check CUDA: {e}")
        return False


def install_diffusers(python_exec: str) -> bool:
    """Check and upgrade diffusers to >= 0.33.0"""
    ColoredOutput.info("Checking diffusers version...")

    current_version = get_package_version("diffusers")
    if current_version:
        ColoredOutput.info(f"Current diffusers version: {current_version}")

        try:
            from packaging import version as pv

            if pv.parse(current_version) >= pv.parse("0.33.0"):
                ColoredOutput.success("diffusers version is >= 0.33.0")
                return True
        except ImportError:
            pass

        ColoredOutput.warning("diffusers version < 0.33.0, upgrading...")
    else:
        ColoredOutput.warning("diffusers not installed, installing...")

    success, output = run_pip(python_exec, "diffusers>=0.33.0")

    if success:
        ColoredOutput.success("diffusers installed/upgraded successfully")
        return True
    else:
        ColoredOutput.error(f"Failed to install diffusers: {output}")
        return False


def install_nvdiffrast(python_exec: str) -> bool:
    """Install nvdiffrast, trying diffrp-nvdiffrast first, then git source"""
    ColoredOutput.info("Installing nvdiffrast...")

    if os.name == "nt":
        ColoredOutput.warning(
            "Windows detected - nvdiffrast may require Visual Studio Build Tools"
        )

    installed_version = get_package_version("diffrp-nvdiffrast")
    if installed_version:
        ColoredOutput.info(f"nvdiffrast already installed: {installed_version}")

        try:
            code = "import nvdiffrast.torch as dr; print('OK')"
            result = subprocess.run(
                [python_exec, "-c", code], capture_output=True, text=True
            )
            if result.returncode == 0:
                ColoredOutput.success("nvdiffrast working correctly")
                return True
        except Exception:
            pass

    ColoredOutput.info(
        "Attempting to install diffrp-nvdiffrast from PyPI (pre-built)..."
    )
    success, output = run_pip(python_exec, "diffrp-nvdiffrast>=0.3.0", capture=True)

    if success:
        ColoredOutput.success("diffrp-nvdiffrast installed successfully")
        return True

    ColoredOutput.warning("PyPI installation failed, trying git source...")

    if os.name == "nt":
        ColoredOutput.info(
            "Installing from source. This requires Visual Studio Build Tools."
        )
        ColoredOutput.info(
            "If this fails, see TROUBLESHOOTING-WINDOWS.md for detailed instructions."
        )

    ColoredOutput.info("Checking for git...")
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        ColoredOutput.success("git found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        ColoredOutput.error("git not found in PATH")
        ColoredOutput.error("Please install git to build nvdiffrast from source")
        return False

    success, output = run_pip(python_exec, "setuptools", "wheel", "ninja", capture=True)

    if not success:
        ColoredOutput.error("Failed to install build dependencies")
        return False

    success, output = run_pip(
        python_exec,
        "git+https://github.com/NVlabs/nvdiffrast.git",
        "--no-build-isolation",
        capture=True,
    )

    if success:
        ColoredOutput.success("nvdiffrast installed from git source successfully")
        return True
    else:
        ColoredOutput.error(f"Failed to install nvdiffrast from git source: {output}")
        if os.name == "nt":
            ColoredOutput.error("\n" + "=" * 60)
            ColoredOutput.error("WINDOWS INSTALLATION FAILED")
            ColoredOutput.error("=" * 60)
            ColoredOutput.error(
                "\nTo fix this, you need to install Visual Studio Build Tools:"
            )
            ColoredOutput.error("1. Download Visual Studio Community 2022")
            ColoredOutput.error(
                "2. During installation, select 'Desktop development with C++'"
            )
            ColoredOutput.error(
                "3. Ensure 'MSVC v143' and 'Windows 10/11 SDK' are checked"
            )
            ColoredOutput.error("4. Restart your computer")
            ColoredOutput.error("5. Run this install script again")
            ColoredOutput.error(
                "\nFor detailed instructions, see: TROUBLESHOOTING-WINDOWS.md"
            )
            ColoredOutput.error("=" * 60 + "\n")
        return False


def install_requirements(python_exec: str, requirements_file: str) -> bool:
    """Install packages from requirements.txt (excluding nvdiffrast)"""
    ColoredOutput.info(f"Installing requirements from {requirements_file}...")

    requirements_path = Path(__file__).parent / requirements_file
    if not requirements_path.exists():
        ColoredOutput.error(f"Requirements file not found: {requirements_path}")
        return False

    requirements_text = requirements_path.read_text()

    requirements = []
    for line in requirements_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "nvdiffrast" not in line.lower():
            requirements.append(line)

    if requirements:
        temp_file = Path(__file__).parent / "requirements_filtered.txt"
        temp_file.write_text("\n".join(requirements))
        success, output = run_pip(python_exec, "-r", str(temp_file))
        temp_file.unlink()
        if success:
            ColoredOutput.success("Core requirements installed successfully")
            return True
        else:
            ColoredOutput.error(f"Failed to install core requirements: {output}")
            return False

    return True


def verify_installation(python_exec: str) -> bool:
    """Verify that critical packages are installed and importable"""
    ColoredOutput.info("Verifying installation...")

    tests = [
        ("numpy", "import numpy as np"),
        ("torch", "import torch"),
        ("diffusers", "from diffusers import AutoencoderKLCosmos"),
        ("PIL", "from PIL import Image"),
        ("safetensors", "import safetensors"),
        ("OpenEXR", "import OpenEXR"),
    ]

    all_passed = True

    for name, import_code in tests:
        try:
            result = subprocess.run(
                [python_exec, "-c", import_code],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                ColoredOutput.success(f"{name} ✓")
            else:
                ColoredOutput.error(f"{name} ✗")
                all_passed = False
        except subprocess.TimeoutExpired:
            ColoredOutput.error(f"{name} ✗ (timeout)")
            all_passed = False
        except Exception as e:
            ColoredOutput.error(f"{name} ✗ ({e})")
            all_passed = False

    try:
        result = subprocess.run(
            [python_exec, "-c", "import nvdiffrast.torch as dr; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            ColoredOutput.success("nvdiffrast ✓")
        else:
            ColoredOutput.warning(
                "nvdiffrast not installed (forward renderer won't work)"
            )
            ColoredOutput.info(
                "See TROUBLESHOOTING-WINDOWS.md for nvdiffrast installation help"
            )
    except (subprocess.TimeoutExpired, Exception) as e:
        ColoredOutput.warning(f"nvdiffrast not available ({e})")
        ColoredOutput.info(
            "See TROUBLESHOOTING-WINDOWS.md for nvdiffrast installation help"
        )

    return all_passed


def main():
    """Main installation function"""
    print("=" * 60)
    print("Materia Installation Script")
    print("=" * 60 + "\n")

    python_exec, env_name = get_python_executable()

    ColoredOutput.info(f"Using Python: {python_exec}")
    ColoredOutput.info(f"Environment: {env_name}\n")

    check_torch_cuda(python_exec)

    steps = [
        (
            "Core requirements",
            lambda: install_requirements(python_exec, "requirements.txt"),
        ),
        ("diffusers >= 0.33.0", lambda: install_diffusers(python_exec)),
        ("nvdiffrast", lambda: install_nvdiffrast(python_exec)),
        ("Verification", lambda: verify_installation(python_exec)),
    ]

    failed_steps = []

    for step_name, step_func in steps:
        print(f"\n{'-' * 60}")
        print(f"Installing {step_name}...")
        print("-" * 60)

        try:
            if not step_func():
                failed_steps.append(step_name)
        except KeyboardInterrupt:
            ColoredOutput.error("\nInstallation interrupted by user")
            sys.exit(1)
        except Exception as e:
            ColoredOutput.error(f"Unexpected error during {step_name}: {e}")
            failed_steps.append(step_name)

    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)

    if not failed_steps:
        ColoredOutput.success("Installation completed successfully!")
        ColoredOutput.info("You can now use Materia in ComfyUI")
    else:
        ColoredOutput.warning(
            f"Installation completed with issues in: {', '.join(failed_steps)}"
        )

        if "nvdiffrast" in failed_steps:
            ColoredOutput.warning("\nNote: The forward renderer requires nvdiffrast.")
            ColoredOutput.warning(
                "The inverse renderer (image to PBR) will work without it."
            )
            ColoredOutput.info("For nvdiffrast help, see: TROUBLESHOOTING-WINDOWS.md")

        if "diffusers >= 0.33.0" in failed_steps:
            ColoredOutput.error(
                "diffusers >= 0.33.0 is REQUIRED. Materia will not work without it."
            )
            ColoredOutput.info("Please upgrade diffusers manually:")
            ColoredOutput.info(f"  {python_exec} -m pip install 'diffusers>=0.33.0'")

    print("=" * 60)

    return 0 if not failed_steps else 1


if __name__ == "__main__":
    sys.exit(main())
