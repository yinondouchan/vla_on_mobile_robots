#!/usr/bin/env python
"""Add the LeRobot v3.0 codebase tag to a Hugging Face dataset repo.

LeRobot resolves dataset versions via git tags (e.g. ``v3.0``). If you uploaded
files manually or skipped ``push_to_hub``, training may fail with a missing
revision error until this tag exists.
"""

from __future__ import annotations

import argparse
import contextlib

from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

# Matches lerobot.datasets.dataset_metadata.CODEBASE_VERSION
DEFAULT_TAG = "v3.0"


def tag_dataset(repo_id: str, revision: str = "main", tag: str = DEFAULT_TAG) -> None:
    api = HfApi()

    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    branch_names = {branch.name for branch in refs.branches}
    if revision not in branch_names:
        raise ValueError(
            f"Revision {revision!r} not found on {repo_id}. "
            f"Available branches: {sorted(branch_names)}"
        )

    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(repo_id, tag=tag, repo_type="dataset")

    api.create_tag(repo_id, tag=tag, revision=revision, repo_type="dataset")
    print(f"Tagged {repo_id}@{revision} as {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the LeRobot v3.0 tag to a Hugging Face dataset."
    )
    parser.add_argument(
        "repo_id",
        help="Dataset repo id, e.g. YinonDouchan/mobile_robot_lift_v1",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or commit to tag (default: main)",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Tag name to create (default: {DEFAULT_TAG})",
    )
    args = parser.parse_args()
    tag_dataset(args.repo_id, revision=args.revision, tag=args.tag)


if __name__ == "__main__":
    main()
