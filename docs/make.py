#!/usr/bin/env python
"""
Script to build documentation for pykokoro.

This script builds the Sphinx documentation for the pykokoro package.
It can be run using:
    python docs/make.py [option]

Options:
    clean   - clean the build directory
    html    - build HTML documentation
    dirhtml - build HTML documentation with directory structure
    all     - build all documentation formats
    help    - show help message
"""

import os
import shutil
import subprocess
import sys


def main():
    """Run the script."""
    sphinx_build = "sphinx-build"

    # Determine if we're being run from docs/ or from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "docs":
        # Running from docs directory
        build_dir = "_build"
        source_dir = "."
    else:
        # Running from project root
        build_dir = os.path.join("docs", "_build")
        source_dir = "docs"

    target = "html" if len(sys.argv) < 2 else sys.argv[1]

    if target == "clean":
        if os.path.exists(build_dir):
            print(f"Cleaning {build_dir}...")
            shutil.rmtree(build_dir)
        return 0

    if target == "help":
        print(__doc__)
        return 0

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Set of valid targets
    valid_targets = {
        "html",
        "dirhtml",
        "latex",
        "latexpdf",
        "text",
        "man",
        "changes",
        "linkcheck",
        "doctest",
        "all",
    }

    if target not in valid_targets:
        print(f"Unknown target: {target}")
        print("Use 'help' target for help")
        return 1

    if target == "all":
        # Build all formats
        for fmt in ["html", "dirhtml", "latex"]:
            cmd = [sphinx_build, "-b", fmt, source_dir, os.path.join(build_dir, fmt)]
            print(f"Building {fmt} documentation...")
            subprocess.run(cmd, check=True)
    else:
        # Build specific format
        cmd = [sphinx_build, "-b", target, source_dir, os.path.join(build_dir, target)]
        print(f"Building {target} documentation...")
        subprocess.run(cmd, check=True)

    print(f"Build finished. Documentation is in {os.path.join(build_dir, target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
