"""Push this repository to the Hugging Face Space.

Run by GitHub Actions on every push to main. It creates the Space on the
first run, uploads the code (uploads handle binary files correctly, unlike
a plain `git push`), removes files that were deleted in GitHub, and keeps
the Space's model settings in sync.

Needs: HF_TOKEN (write token) and HF_SPACE ("username/space-name").
Optional: HSD_MODEL_REPO ("username/model-repo") to point the app at the weights.
"""

import os
import sys

from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
space = os.environ.get("HF_SPACE")
if not token or not space:
    sys.exit("HF_TOKEN and HF_SPACE must be set (GitHub secret + variable).")

api = HfApi(token=token)
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

api.upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=".",
    commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local')[:7]} from GitHub",
    ignore_patterns=[".git/*", ".github/*", "tests/*", "scripts/*", "*.pt", "*.png", "venv/*", "**/__pycache__/*"],
    delete_patterns=["*.py", "**/*.py", "*.txt", "results/*", ".streamlit/*", "Dockerfile", "README.md"],
)

model_repo = os.environ.get("HSD_MODEL_REPO")
if model_repo:
    api.add_space_variable(space, "HSD_MODEL_REPO", model_repo)
    # the app needs a token to read a private model repo
    api.add_space_secret(space, "HF_TOKEN", token)

print(f"Deployed to https://huggingface.co/spaces/{space}")
