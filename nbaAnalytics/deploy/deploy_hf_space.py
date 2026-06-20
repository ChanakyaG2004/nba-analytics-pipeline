import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, whoami


def deploy(args):
    user = whoami()["name"]
    repo_id = args.repo_id or f"{user}/nba-ai-analytics-dashboard"
    space_dir = Path(args.space_dir)

    if not space_dir.exists():
        raise FileNotFoundError(f"Space directory not found: {space_dir}")

    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=space_dir,
        path_in_repo=".",
        commit_message="Deploy NBA analytics dashboard",
        delete_patterns=["__pycache__/*", "*.pyc"],
    )
    return repo_id


def build_parser():
    parser = argparse.ArgumentParser(description="Deploy the bundled demo to Hugging Face Spaces.")
    parser.add_argument("--repo-id", help="Optional repo id, e.g. username/nba-ai-analytics-dashboard.")
    parser.add_argument("--space-dir", default="hf_space")
    parser.add_argument("--private", action="store_true")
    return parser


if __name__ == "__main__":
    repo_id = deploy(build_parser().parse_args())
    print(f"https://huggingface.co/spaces/{repo_id}")
