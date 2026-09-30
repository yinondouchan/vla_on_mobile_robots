import numpy as np
import pygame


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
