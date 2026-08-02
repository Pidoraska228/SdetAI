#!/usr/bin/env python3
"""
Build and run script for SdetAI.
Works on Windows (with MSVC) and Linux (with GCC/Clang).
"""
import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

ROOT = Path(__file__).parent.absolute()
BUILD_DIR = ROOT / "build"

def run(cmd, cwd=None, check=True):
    """Run command and return result."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result

def detect_generator():
    """Detect best CMake generator for platform."""
    if shutil.which("ninja"):
        return "Ninja"
    system = platform.system()
    if system == "Windows":
        vswhere = shutil.which("vswhere")
        if vswhere:
            return "Visual Studio 17 2022"
        return "NMake Makefiles"
    return "Unix Makefiles"

def configure():
    """Configure CMake build."""
    BUILD_DIR.mkdir(exist_ok=True)
    
    generator = detect_generator()
    config = [
        "cmake",
        f'-G"{generator}"',
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_STANDARD=20",
        f'"{ROOT}"'
    ]
    
    if platform.system() == "Windows" and "Visual Studio" in generator:
        config.insert(1, "-A")
        config.insert(2, "x64")
    
    run(" ".join(config), cwd=BUILD_DIR)

def build(target=None, parallel=True):
    """Build project."""
    cmd = ["cmake", "--build", str(BUILD_DIR), "--config", "Release"]
    if target:
        cmd.extend(["--target", target])
    if parallel:
        cmd.extend(["--parallel", str(os.cpu_count() or 4)])
    run(" ".join(cmd))

def clean():
    """Clean build directory."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    print("Cleaned build directory")

def download_model():
    """Download a small GGUF model for testing."""
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    # TinyLlama 1.1B Q4_K_M - ~600MB, good for coding
    model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_path = models_dir / "tinyllama-1.1b-chat.Q4_K_M.gguf"
    
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return
    
    print(f"Downloading model to {model_path}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Manual download: wget -O models/tinyllama.gguf", model_url)

def run_agent(workspace=None, model=None, task=None):
    """Run the agent."""
    exe = BUILD_DIR / "Release" / "sdetai_main.exe"
    if not exe.exists():
        exe = BUILD_DIR / "sdetai_main"
    if not exe.exists():
        print("Executable not found. Run build first.")
        return
    
    cmd = [str(exe)]
    if workspace:
        cmd.extend(["--workspace", workspace])
    if model:
        cmd.extend(["--model", model])
    if task:
        cmd.extend(["--task", task])
    
    run(" ".join(cmd), check=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SdetAI build & run")
    parser.add_argument("command", nargs="?", default="build", 
                        choices=["configure", "build", "clean", "rebuild", 
                                 "download-model", "run", "test"])
    parser.add_argument("--workspace", default=str(ROOT / "test_workspace"))
    parser.add_argument("--model", default=str(ROOT / "models" / "tinyllama-1.1b-chat.Q4_K_M.gguf"))
    parser.add_argument("--task", default="Create a simple C++ hello world")
    args = parser.parse_args()
    
    if args.command == "configure":
        configure()
    elif args.command == "build":
        build()
    elif args.command == "clean":
        clean()
    elif args.command == "rebuild":
        clean()
        configure()
        build()
    elif args.command == "download-model":
        download_model()
    elif args.command == "run":
        run_agent(args.workspace, args.model, args.task)
    elif args.command == "test":
        build(target="test")
        # Run CTest
        run("ctest --output-on-failure", cwd=BUILD_DIR)

if __name__ == "__main__":
    main()