import numpy as np
import pygame

from vlamobile.inference import LerobotInference, LerobotInferenceWithPlanner
# from lerobot.policies.rtc import RTCConfig, ActionQueue
# from lerobot.configs import RTCAttentionSchedule


class JoystickControl:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        else:
            joystick = None

        self.joystick = joystick

        self.forward = 0.0
        self.turn = 0.0
        self.lift = 0.0
        self.grip = 0.0

    def __call__(self, state, frames, task: str):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 4:
                    self.forward = -event.value
                elif event.axis == 3:
                    self.turn = -event.value
                elif event.axis == 1:
                    self.lift = event.value
                elif event.axis == 2:
                    # Left trigger: typically -1 (released) .. +1 (pressed) -> [0, 1]
                    self.grip = (event.value + 1.0) * 0.5
                    # print(f"Axis {event.axis} moved to {event.value:.2f}")

                # elif event.type == pygame.JOYBUTTONUP:
                #     print(f"Button {event.button} released")

                # elif event.type == pygame.JOYHATMOTION:
                #     print(f"Hat/D-pad {event.hat} moved to {event.value}")

        return np.array(
            [self.forward, self.turn, self.grip, self.lift], dtype=np.float64
        )

    def reset(self):
        self.forward = 0.0
        self.turn = 0.0
        self.lift = 0.0
        self.grip = 0.0


class PolicyControl:
    def __init__(self, policy_path, dataset_repo_id, use_policy_planner: bool = False):
        if use_policy_planner:
            self.policy = LerobotInferenceWithPlanner(
                policy_path=policy_path,
                dataset_repo_id=dataset_repo_id,
                robot_type="custom_mobile_robot",
            )
        else:
            self.policy = LerobotInference(
                policy_path=policy_path,
                dataset_repo_id=dataset_repo_id,
                robot_type="custom_mobile_robot",
            )

    def __call__(self, state, frames, task: str):
        action = self.policy.predict(state=state, images=frames, task=task)
        return action

    def reset(self):
        self.policy.reset()


# class AsynchronousPolicyControl:
#       def __init__(self, policy_path, dataset_repo_id):
#         self.policy = LerobotInference(
#           policy_path=policy_path,
#           dataset_repo_id=dataset_repo_id,
#           robot_type="custom_mobile_robot",
#         )
#         self.policy.reset()
#         self.policy.cfg.rtc_config = RTCConfig(
#           enabled=True,
#           execution_horizon=10,  # How many steps to blend with previous chunk
#           max_guidance_weight=10.0,  # How strongly to enforce consistency
#           prefix_attention_schedule=RTCAttentionSchedule.EXP,  # Exponential blend
#         )
#         self.action_queue = ActionQueue(self.policy.cfg.rtc_config)


#       def __call__(self, state, frames, task: str):
#         action = self.policy.predict(state=state, images=frames, task=task)
#         return action

#       def reset(self):
#           self.policy.reset()
