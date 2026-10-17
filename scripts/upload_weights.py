"""One-time: upload student_best.pt to a (private) Hugging Face model repo.

Usage (from the folder that contains student_best.pt):
    pip install huggingface_hub
    huggingface-cli login            # paste a WRITE token
    python scripts/upload_weights.py sanjida-khanom/hate-speech-distilled-bert
"""

import sys

from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="student_best.pt", path_in_repo="student_best.pt",
                repo_id=repo, repo_type="model", commit_message="Upload distilled student weights")
print(f"Uploaded to https://huggingface.co/{repo}")
