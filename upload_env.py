from huggingface_hub import HfApi
api = HfApi()
repo = "YinonDouchan/mobile_robot_lift_env"
files = [
    "env.py",
    "simple_scene.xml",
    "simple_custom_robot.xml",
    "assets/gear_50.stl",
    "assets/gripper_9g/Claw.stl",
    "assets/gripper_9g/Base.stl",
    "assets/gripper_9g/Connector_Gear.stl",
    "assets/gripper_9g/Connector.stl",
    "assets/lift_scene/textures/light-gray-floor-tile.png",
    "assets/lift_scene/textures/red-wood.png",
]
for path in files:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path,
        repo_id=repo,
        repo_type="model",
    )