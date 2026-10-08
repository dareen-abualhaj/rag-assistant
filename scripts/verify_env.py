"""Verify that the environment is ready: GPU availability + core imports.

Usage:
    python scripts/verify_env.py              # strict: requires CUDA
    python scripts/verify_env.py --allow-cpu  # local machines without NVIDIA GPU
"""
import argparse
import importlib
import sys


def check_cuda(allow_cpu: bool):
    import torch

    available = torch.cuda.is_available()
    if available:
        print(f"[OK] CUDA available | device: {torch.cuda.get_device_name(0)}")
        print(f"     torch {torch.__version__} | CUDA {torch.version.cuda}")
        return
    if allow_cpu:
        print("[WARN] CUDA not available, continuing on CPU (--allow-cpu).")
        return
    raise AssertionError(
        "CUDA/GPU is NOT available. Use a machine with an NVIDIA GPU "
        "(or Google Colab / Kaggle), or run with --allow-cpu."
    )


def check_imports():
    modules = ["pandas", "sentence_transformers", "chromadb", "spacy"]
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            raise AssertionError(f"Failed to import '{name}': {e}")
        print(f"[OK] import {name} ({getattr(mod, '__version__', 'unknown')})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}")
    check_imports()
    check_cuda(args.allow_cpu)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
