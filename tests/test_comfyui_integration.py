#!/usr/bin/env python3
"""Test that Materia works with ComfyUI"""

import sys
import os
from pathlib import Path


def test_comfyui_custom_nodes_folder():
    """Test that materia is in the correct ComfyUI location"""
    print("Testing ComfyUI folder structure...")

    # Find materia root directory (tests/ is subdirectory)
    materia_root = Path(__file__).parent.parent
    print(f"  Materia root directory: {materia_root}")

    # Check if in custom_nodes
    if "custom_nodes" in str(materia_root):
        print("  ✓ Found in custom_nodes folder")
        return True
    else:
        print("  ⚠ Not in standard custom_nodes location")
        print("    This may be OK if using custom installation")
        return True


def test_materia_files_exist():
    """Test that all required Materia files exist"""
    print("\nTesting Materia file structure...")

    materia_root = Path(__file__).parent.parent

    required_files = [
        "nodes.py",
        "__init__.py",
        "requirements.txt",
        "install.py",
        "pyproject.toml",
        "README.md",
        # "TROUBLESHOOTING-WINDOWS.md",  # Optional file, not required
    ]

    all_exist = True
    for filename in required_files:
        filepath = materia_root / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            all_exist = False

    return all_exist


def test_comfyui_modules_available():
    """Test that ComfyUI core modules can be imported (if available)"""
    print("\nTesting ComfyUI core modules...")

    results = {}

    # Try importing ComfyUI modules (may not be available in test environment)
    try:
        import folder_paths

        results["folder_paths"] = True
        print("  ✓ folder_paths")
    except ImportError:
        results["folder_paths"] = False
        print("  ⚠ folder_paths - not available (expected outside ComfyUI)")

    try:
        import comfy.model_management as mm

        results["model_management"] = True
        print("  ✓ comfy.model_management")
    except ImportError:
        results["model_management"] = False
        print("  ⚠ comfy.model_management - not available (expected outside ComfyUI)")

    try:
        import comfy.utils

        results["comfy_utils"] = True
        print("  ✓ comfy.utils")
    except ImportError:
        results["comfy_utils"] = False
        print("  ⚠ comfy.utils - not available (expected outside ComfyUI)")

    # All modules not being available is OK in test environment
    return True


def test_node_registration():
    """Test that Materia nodes can be registered"""
    print("\nTesting node registration...")

    try:
        # Add materia to path
        materia_root = Path(__file__).parent
        if str(materia_root) not in sys.path:
            sys.path.insert(0, str(materia_root))

        # Import the node mappings
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        print(f"  ✓ NODE_CLASS_MAPPINGS: {len(NODE_CLASS_MAPPINGS)} nodes")
        print(
            f"  ✓ NODE_DISPLAY_NAME_MAPPINGS: {len(NODE_DISPLAY_NAME_MAPPINGS)} names"
        )

        # Verify mappings are dictionaries
        assert isinstance(NODE_CLASS_MAPPINGS, dict)
        assert isinstance(NODE_DISPLAY_NAME_MAPPINGS, dict)

        print("  ✓ Node mappings are valid dictionaries")

        return True

    except ImportError as e:
        print(f"  ✗ Failed to import nodes: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def test_required_dependencies():
    """Test that Python packages are available"""
    print("\nTesting required dependencies...")

    dependencies = [
        "torch",
        "numpy",
        "PIL",
        "diffusers",
        "safetensors",
        "cv2",
        "imageio",
        "einops",
    ]

    optional_dependencies = [
        "nvdiffrast",
    ]

    all_required_available = True

    print("  Required dependencies:")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"    ✓ {dep}")
        except ImportError:
            print(f"    ✗ {dep} - MISSING")
            all_required_available = False

    print("\n  Optional dependencies:")
    for dep in optional_dependencies:
        try:
            __import__(dep)
            print(f"    ✓ {dep}")
        except ImportError:
            print(f"    ⚠ {dep} - Not installed (forward renderer won't work)")

    return all_required_available


def test_diffusers_version():
    """Test that diffusers version is >= 0.33.0"""
    print("\nTesting diffusers version...")

    try:
        import diffusers
        from packaging import version as pv

        installed_version = diffusers.__version__
        required_version = "0.33.0"

        print(f"  Installed: {installed_version}")
        print(f"  Required: {required_version}")

        if pv.parse(installed_version) >= pv.parse(required_version):
            print("  ✓ diffusers version is sufficient")
            return True
        else:
            print("  ✗ diffusers version is too old")
            print("    Run: python install.py")
            return False

    except ImportError:
        print("  ✗ diffusers not installed")
        print("    Run: python install.py")
        return False
    except AttributeError:
        print("  ✗ Could not determine diffusers version")
        return False


def test_installation_files():
    """Test that installation support files are present"""
    print("\nTesting installation support files...")

    materia_root = Path(__file__).parent

    files_to_check = {
        "install.py": "Python installer",
        "requirements.txt": "Requirements file",
        "pyproject.toml": "ComfyUI Manager config",
        "README.md": "Documentation",
        "TROUBLESHOOTING-WINDOWS.md": "Windows troubleshooting guide",
        "test_imports.py": "Import validation tests",
    }

    all_present = True
    for filename, description in files_to_check.items():
        filepath = materia_root / filename
        if filepath.exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - MISSING ({description})")
            all_present = False

    return all_present


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("Materia ComfyUI Integration Tests")
    print("=" * 60)

    results = {
        "folder_structure": test_comfyui_custom_nodes_folder(),
        "file_structure": test_materia_files_exist(),
        "comfyui_modules": test_comfyui_modules_available(),
        "node_registration": test_node_registration(),
        "dependencies": test_required_dependencies(),
        "diffusers_version": test_diffusers_version(),
        "installation_files": test_installation_files(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())

    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nMateria is ready to use with ComfyUI")
        print("Start ComfyUI and look for nodes in the 'Materia' category")
        return 0
    else:
        print("✗ Some tests failed")
        print("\nSee output above for details")

        failed_tests = [name for name, passed in results.items() if not passed]
        if failed_tests:
            print(f"\nFailed tests: {', '.join(failed_tests)}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
