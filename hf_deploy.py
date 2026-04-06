import os
import sys

# Ensure huggingface_hub is installed
try:
    from huggingface_hub import HfApi
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import HfApi

token = os.getenv("HF_TOKEN")
if not token:
    print("Error: HF_TOKEN environment variable not set. Please set it before running this script.")
    sys.exit(1)
api = HfApi(token=token)
repo_id = "GuptaAnubhav/trustops-env"

print(f"Creating Space {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

print(f"Uploading files from trustops-env...")
api.upload_folder(
    folder_path="/Users/anubhavgupta/Desktop/Scaler1/trustops-env",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc"]
)

print("Deploy successful! View your space at: https://huggingface.co/spaces/" + repo_id)
