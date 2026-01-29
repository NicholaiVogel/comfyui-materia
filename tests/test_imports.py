#!/usr/bin/env python3
"""Test that all critical imports work as expected for Materia"""

import sys
from pathlib import Path

# Add materia root to path (tests/ is subdirectory)
materia_root = Path(__file__).parent.parent
if str(materia_root) not in sys.path:
    sys.path.insert(0, str(materia_root))


def test_diffusers_imports():
    """Test diffusers-specific imports"""
    print("Testing diffusers imports...")

    direct_import_works = False
    submodule_import_works = False

    try:
        from diffusers import AutoencoderKLCosmos

        print("  ✓ Direct import from diffusers works")
        direct_import_works = True
    except ImportError as e:
        print(f"  ✗ Direct import failed: {e}")

    try:
        from diffusers.models.autoencoders.autoencoder_kl_cosmos import (
            AutoencoderKLCosmos,
        )

        print("  ✓ Import from submodule works")
        submodule_import_works = True

        if not direct_import_works:
            print("\n  ⚠ WARNING: nodes.py uses direct import path!")
            print("  ⚠ Update nodes.py line 17 to use submodule import path")
    except ImportError as e:
        print(f"  ✗ Submodule import failed: {e}")

    if not direct_import_works and not submodule_import_works:
        print("\n  ✗ FAILED: AutoencoderKLCosmos not found in either location")
        print("  ✗ Check diffusers version (requires >= 0.33.0)")
        return False

    return True


def test_nvdiffrast_import():
    """Test nvdiffrast import"""
    print("\nTesting nvdiffrast import...")

    try:
        import nvdiffrast.torch as dr

        print("  ✓ nvdiffrast imports correctly")
        return True
    except ImportError as e:
        print(f"  ✗ nvdiffrast import failed: {e}")
        print("\n  Install nvdiffrast using:")
        print("    pip install diffrp-nvdiffrast")
        print("  Or build from source:")
        print("    pip install git+https://github.com/NVlabs/nvdiffrast.git")
        return False


def test_basic_dependencies():
    """Test basic dependency imports"""
    print("\nTesting basic dependencies...")

    tests = [
        ("numpy", "import numpy as np"),
        ("torch", "import torch"),
        ("PIL", "from PIL import Image"),
        ("safetensors", "import safetensors"),
        ("cv2", "import cv2"),
        ("imageio", "import imageio"),
        ("einops", "import einops"),
        ("OpenEXR", "import OpenEXR"),
    ]

    all_passed = True
    for name, import_code in tests:
        try:
            exec(import_code)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            all_passed = False

    return all_passed


def test_materia_node_imports():
    """Test that Materia nodes can be imported"""
    print("\nTesting Materia nodes...")

    try:
        from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        node_count = len(NODE_CLASS_MAPPINGS)
        print(f"  ✓ Nodes loaded successfully: {node_count} node classes")
        print(f"  ✓ Display names: {len(NODE_DISPLAY_NAME_MAPPINGS)} entries")

        # Check for expected node classes
        expected_nodes = [
            "LoadDiffusionRendererModel",
            "InverseRenderingNode",
            "ForwardRenderingNode",
            "LoadHDRImage",
        ]

        found_nodes = []
        for expected in expected_nodes:
            for node_name in NODE_CLASS_MAPPINGS.keys():
                if expected in str(node_name):
                    found_nodes.append(expected)
                    break

        print(
            f"\n  Expected node classes found: {len(found_nodes)}/{len(expected_nodes)}"
        )
        for node in found_nodes:
            print(f"    ✓ {node}")

        missing = set(expected_nodes) - set(found_nodes)
        if missing:
            print(f"    ✗ Missing nodes: {missing}")

        return len(missing) == 0

    except ImportError as e:
        print(f"  ✗ Node import failed: {e}")
        print("\n  Check that all dependencies are installed")
        print("  Run: python install.py")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error importing nodes: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all import tests"""
    print("=" * 60)
    print("Materia Import Validation Tests")
    print("=" * 60)

    results = {
        "diffusers": test_diffusers_imports(),
        "nvdiffrast": test_nvdiffrast_import(),
        "basic_deps": test_basic_dependencies(),
        "materia_nodes": test_materia_node_imports(),
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
        return 0
    else:
        print("✗ Some tests failed")
        print("\nSee output above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
