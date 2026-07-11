import fire
from huggingface_hub import HfApi

def main(repo, env_path="vlamobile/env.py"):
    api = HfApi()
    files = [
        (env_path, "env.py"),
        ("vlamobile/lift_env.py", "lift_env.py"),
        "assets/mobile_robot_lift/scene.xml",
        "assets/mobile_robot_lift/robot.xml",
        "assets/mobile_robot_lift/textures/light-gray-floor-tile.png",
        "assets/mobile_robot_lift/textures/red-wood.png",
    ]
    for entry in files:
        if isinstance(entry, tuple):
            local_path, repo_path = entry
        else:
            local_path = repo_path = entry
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="model",
        )

if __name__ == "__main__":
    fire.Fire(main)