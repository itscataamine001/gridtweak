#!/usr/bin/env python3
"""
GridTweak - Render Deployment Assistant
----------------------------------------
This script prepares your project for deployment on Render.
It generates a render.yaml blueprint and optionally initialises
a Git repository and pushes to GitHub.

Usage:
    python deploy_to_render.py [--remote <github-remote-url>]

If --remote is given, the script will init git, add the remote,
commit and push. Otherwise it just writes render.yaml.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# ------------------------------
# 1.  Required files
# ------------------------------
REQUIRED_FILES = [
    "dlr_v28.py",
    "config.json",
    "requirements.txt",
    "lgb_best.pkl",          # if using ML
    "lgb_best_scaler.pkl"    # if using ML
]

def check_files():
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print("❌ The following required files are missing:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease make sure all files are in the current directory.")
        return False
    print("✅ All required files found.")
    return True

# ------------------------------
# 2.  Generate render.yaml
# ------------------------------
def generate_render_yaml(config_path="config.json"):
    """Reads config.json and creates a render.yaml blueprint."""
    # Load config to get some values (optional)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        config = {}

    # Use default values if not present
    service_name = "gridtweak"
    port = 8000

    # Build command
    build_cmd = "pip install -r requirements.txt"

    # Start command: use $PORT from environment
    start_cmd = "python dlr_v28.py --api --config config.json --port $PORT"

    # Health check path (root endpoint returns json)
    health_check = "/"

    # Environment variables (optional)
    env_vars = {
        "PYTHONUNBUFFERED": "1"
    }

    render_yaml = f"""
services:
  - type: web
    name: {service_name}
    runtime: python
    plan: free
    buildCommand: {build_cmd}
    startCommand: {start_cmd}
    healthCheckPath: {health_check}
    envVars:
"""
    for key, value in env_vars.items():
        render_yaml += f"      - key: {key}\n        value: {value}\n"

    # Add any extra env vars from config (if they are secrets, but we don't have any)
    # If you have email/slack credentials, they should be set via Render's UI or env vars.

    # Write to file
    with open("render.yaml", "w") as f:
        f.write(render_yaml)

    print("✅ render.yaml generated successfully.")
    return True

# ------------------------------
# 3.  Git setup and push
# ------------------------------
def git_push(remote_url):
    """Initialize git repo, add remote, commit and push."""
    # Check if git is installed
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("❌ Git is not installed or not in PATH. Skipping Git operations.")
        return False

    # Check if already a git repo
    if Path(".git").exists():
        print("ℹ️  Git repository already exists. Skipping init.")
    else:
        print("🔧 Initializing Git repository...")
        subprocess.run(["git", "init"], check=True)

    # Add remote if provided and not already present
    if remote_url:
        # Check if remote already exists
        result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == remote_url:
            print("ℹ️  Remote 'origin' already points to the given URL.")
        else:
            if result.returncode == 0:
                subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True)
            else:
                subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
            print(f"✅ Remote 'origin' set to {remote_url}")

    # Add all files
    subprocess.run(["git", "add", "."], check=True)

    # Commit if there are changes
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        print("📝 Committing changes...")
        subprocess.run(["git", "commit", "-m", "Deploy GridTweak V28 via Render"], check=True)
    else:
        print("ℹ️  No changes to commit.")

    # Push to remote if remote exists
    if remote_url:
        print("🚀 Pushing to remote...")
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("✅ Push completed.")
    else:
        print("ℹ️  No remote URL provided – skipping push.")

    return True

# ------------------------------
# 4.  Final instructions
# ------------------------------
def print_instructions(remote_url=None):
    print("\n" + "="*70)
    print("🎉 DEPLOYMENT PREPARATION COMPLETE")
    print("="*70)
    if remote_url:
        print(f"📦 Repository pushed to: {remote_url}")
        print("\nNow go to Render and create a new Web Service from this repository.")
        print("The render.yaml will be automatically detected.")
    else:
        print("📄 The render.yaml blueprint has been created.")
        print("\nNext steps:")
        print("1. Push your code to a Git repository (GitHub, GitLab, etc.)")
        print("2. On Render, click 'New +' → 'Blueprint' and connect your repo.")
        print("3. Render will pick up render.yaml and deploy your app.")

    print("\n🔗 After deployment, your dashboard will be available at:")
    print("   https://your-service-name.onrender.com/dashboard")
    print("\nDon't forget to set your custom domain in Render's settings.")
    print("="*70)

# ------------------------------
# 5.  Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="GridTweak Render Deployment Helper")
    parser.add_argument("--remote", type=str, help="GitHub remote URL (e.g., https://github.com/username/repo.git)")
    args = parser.parse_args()

    # Check files
    if not check_files():
        sys.exit(1)

    # Generate render.yaml
    if not generate_render_yaml():
        sys.exit(1)

    # Git operations if remote provided
    if args.remote:
        git_push(args.remote)

    # Print final instructions
    print_instructions(args.remote)

if __name__ == "__main__":
    main()