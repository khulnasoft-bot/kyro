#!/usr/bin/env python3
"""Run Kyro tutorials.

Usage:
    python scripts/run_tutorial.py 01_pattern_matching
    python scripts/run_tutorial.py --list
    python scripts/run_tutorial.py --all
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TUTORIALS_DIR = Path(__file__).parent.parent / "tutorials"

def list_tutorials():
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    for nb in notebooks:
        print(f"  {nb.stem}")
    md_files = sorted((TUTORIALS_DIR / "docs").glob("*.md")) if (TUTORIALS_DIR / "docs").exists() else []
    for md in md_files:
        print(f"  {md.stem}")

def run_notebook(tutorial_name):
    nb_path = TUTORIALS_DIR / f"{tutorial_name}.ipynb"
    if not nb_path.exists():
        print(f"Error: {nb_path} not found")
        sys.exit(1)
    print(f"Running {tutorial_name}...")
    subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", str(nb_path)], check=True)
    print(f"✓ {tutorial_name} executed successfully")

def run_all():
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    for nb in notebooks:
        try:
            subprocess.run(["jupyter", "nbconvert", "--execute", "--inplace", str(nb)], check=True)
            print(f"✓ {nb.stem}")
        except subprocess.CalledProcessError:
            print(f"✗ {nb.stem} FAILED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kyro tutorials")
    parser.add_argument("tutorial", nargs="?", help="Tutorial name (e.g., 01_pattern_matching)")
    parser.add_argument("--list", action="store_true", help="List all tutorials")
    parser.add_argument("--all", action="store_true", help="Run all tutorials")
    args = parser.parse_args()

    if args.list:
        list_tutorials()
    elif args.all:
        run_all()
    elif args.tutorial:
        run_notebook(args.tutorial)
    else:
        parser.print_help()
