#!/usr/bin/env python3
"""
SC-SINDy Colab Setup Script

Run this in a Jupyter/Colab cell to clone or update the repository.

Usage:
    # In a Colab cell, run:
    !wget -q https://raw.githubusercontent.com/YOUR_USERNAME/structure-constrained-sindy/main/scripts/setup_colab.py
    %run setup_colab.py

    # Or just copy-paste this entire file into a cell and run it.
"""

import os
import subprocess

# Configuration - UPDATE THIS WITH YOUR REPO URL
REPO_URL = "https://github.com/ethanwfang/structured-contained-sindy.git"
REPO_DIR = "structure-contained-sindy"
BRANCH = "main"


def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def setup_repository():
    """Clone or update the repository."""
    print("=" * 60)
    print("SC-SINDy Colab Setup")
    print("=" * 60)

    if os.path.exists(REPO_DIR):
        print(f"\nRepository already exists at ./{REPO_DIR}")
        print("Pulling latest changes...")

        # Pull latest changes
        code, out, err = run_command(f"git pull origin {BRANCH}", cwd=REPO_DIR)

        if code == 0:
            print("Successfully updated!")
            if out.strip():
                print(out)
        else:
            print(f"Warning: git pull failed: {err}")
            print("Trying to reset to origin...")
            run_command(f"git fetch origin && git reset --hard origin/{BRANCH}", cwd=REPO_DIR)
    else:
        print(f"\nCloning repository...")
        print(f"URL: {REPO_URL}")

        code, out, err = run_command(f"git clone {REPO_URL}")

        if code == 0:
            print("Successfully cloned!")
        else:
            print(f"Error cloning: {err}")
            return False

    # Change to repo directory
    os.chdir(REPO_DIR)
    print(f"\nCurrent directory: {os.getcwd()}")

    # Install the package
    print("\nInstalling SC-SINDy package...")
    code, out, err = run_command("pip install -e '.[torch,viz]' -q")

    if code == 0:
        print("Installation complete!")
    else:
        print(f"Installation warning: {err}")

    # Verify installation
    print("\nVerifying installation...")
    try:
        import sc_sindy
        print(f"SC-SINDy version: {sc_sindy.__version__}")
        print("Installation verified!")
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    print("\n" + "=" * 60)
    print("Setup complete! You can now run:")
    print("  !python scripts/run_tests.py")
    print("  !python scripts/run_fair_evaluation.py")
    print("=" * 60)

    return True


if __name__ == "__main__":
    setup_repository()
