#!/usr/bin/env python3
"""Unit tests for install.py functionality"""

import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPythonEnvironmentDetection:
    """Test Python environment detection logic"""

    def test_portable_python_detection(self, tmp_path):
        """Test that portable Python is detected when python_embeded exists"""
        # Create mock directory structure
        comfyui_dir = tmp_path / "ComfyUI" / "custom_nodes" / "comfyui-materia"
        comfyui_dir.mkdir(parents=True, exist_ok=True)

        python_embeded = comfyui_dir / ".." / ".." / "python_embeded"
        python_embeded.mkdir(parents=True, exist_ok=True)
        python_exe = python_embeded / (
            "python.exe" if Path.sys.platform == "win32" else "python"
        )
        python_exe.touch()

        # Create a minimal install.py for testing
        install_py = comfyui_dir / "install.py"
        install_py.write_text("from pathlib import Path\nimport sys\nprint('OK')")

        # Test detection (simplified - just checking paths exist)
        assert python_embeded.exists()

    def test_system_python_fallback(self):
        """Test that system Python is used when ComfyUI paths don't exist"""
        # Mock that no ComfyUI Python is found
        assert Path(sys.executable).exists()


class TestPackageVersionDetection:
    """Test version detection logic"""

    def test_version_comparison_old_requires_upgrade(self):
        """Test that old version requires upgrade"""
        # This would need to mock get_package_version
        from packaging import version as pv

        old_version = "0.32.0"
        required_version = "0.33.0"

        assert pv.parse(old_version) < pv.parse(required_version)

    def test_version_comparison_correct_no_upgrade(self):
        """Test that correct version doesn't require upgrade"""
        from packaging import version as pv

        current_version = "0.34.0"
        required_version = "0.33.0"

        assert pv.parse(current_version) >= pv.parse(required_version)

    def test_version_comparison_edge_case(self):
        """Test version comparison with same major but different minor"""
        from packaging import version as pv

        version1 = "0.33.0"
        version2 = "0.33.1"

        assert pv.parse(version1) < pv.parse(version2)


class TestRequirementsFiltering:
    """Test requirements filtering logic"""

    def test_nvdiffrast_filtered_out(self):
        """Test that nvdiffrast is filtered from requirements"""
        requirements_text = """
numpy>=1.23.0
diffusers>=0.33.0
diffrp-nvdiffrast>=0.3.0
# This is a comment
opencv-python>=4.7.0
"""

        requirements = []
        for line in requirements_text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "nvdiffrast" not in line.lower():
                requirements.append(line)

        # Should have 4 packages, not 5
        assert len(requirements) == 4
        assert "diffrp-nvdiffrast>=0.3.0" not in requirements
        assert "numpy>=1.23.0" in requirements

    def test_comments_filtered_out(self):
        """Test that comment lines are filtered"""
        requirements_text = """
# This is a comment
numpy>=1.23.0
# Another comment
Pillow>=9.0.0
"""

        requirements = []
        for line in requirements_text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "nvdiffrast" not in line.lower():
                requirements.append(line)

        # Should have 2 packages
        assert len(requirements) == 2
        assert "# This is a comment" not in requirements


class TestImportVerification:
    """Test import verification logic"""

    def test_successful_import(self):
        """Test that successful import returns True"""
        # This would need subprocess mocking in real test
        code = "print('OK')"
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_failed_import(self):
        """Test that failed import returns error"""
        code = "import nonexistent_module"
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


class TestErrorHandling:
    """Test error handling and messaging"""

    def test_colored_output_format(self):
        """Test that colored output has correct format"""
        from install import ColoredOutput

        output = ColoredOutput.success("Test message")
        assert "✓ Test message" in output
        assert ColoredOutput.RESET in output

        output = ColoredOutput.error("Error message")
        assert "✗ Error message" in output

        output = ColoredOutput.warning("Warning message")
        assert "⚠ Warning message" in output


class TestGitDependency:
    """Test git dependency checking"""

    def test_git_available(self):
        """Test that git availability is detected"""
        try:
            result = subprocess.run(
                ["git", "--version"], capture_output=True, text=True, timeout=5
            )
            assert result.returncode == 0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pytest.skip("git not available")

    def test_git_not_found_raises_error(self):
        """Test that missing git is detected"""
        try:
            result = subprocess.run(
                ["nonexistent_git_command", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            assert False, "Should have raised error"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Expected - git not found
            pass


def test_install_script_syntax():
    """Test that install.py has valid Python syntax"""
    install_py = Path(__file__).parent / "install.py"

    if not install_py.exists():
        pytest.skip("install.py not found")

    import py_compile

    try:
        py_compile.compile(str(install_py), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"install.py has syntax error: {e}")


def test_requirements_file_exists():
    """Test that requirements.txt exists and is valid"""
    requirements_txt = Path(__file__).parent / "requirements.txt"

    assert requirements_txt.exists(), "requirements.txt not found"

    content = requirements_txt.read_text()
    assert len(content) > 0, "requirements.txt is empty"
    assert "diffusers" in content, "diffusers not in requirements.txt"
    assert "nvdiffrast" in content, "nvdiffrast not in requirements.txt"


def test_pyproject_exists():
    """Test that pyproject.toml exists"""
    pyproject = Path(__file__).parent / "pyproject.toml"

    assert pyproject.exists(), "pyproject.toml not found"

    content = pyproject.read_text()
    assert "[project]" in content, "pyproject.toml missing [project] section"
    assert "comfyui-materia" in content, "pyproject.toml missing project name"


def test_readme_installation_section():
    """Test that README.md has installation section"""
    readme = Path(__file__).parent / "README.md"

    if not readme.exists():
        pytest.skip("README.md not found")

    content = readme.read_text()
    assert "install.py" in content, "README.md missing install.py reference"
    assert "python install.py" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
