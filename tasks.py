import random
import numpy as np


TASKS = [
  f"Pick up the small cube",
  "Pick up the medium cube",
  "Pick up the large cube",
  "Lift the small cube three inches above the surface",
  "Touch the red platform with the claw",
  "Move the claw directly above the green platform",
  "Open the claw while holding nothing over the blue platform",
  "Touch the top face of the large cube",
  "Nudge the medium cube to the left",
  "Hover the claw exactly two inches above the small cube",
  "Place the small cube on the red platform",         
  "Place the medium cube on the blue platform",
  "Place the large cube on the green platform",
  "Move the small cube from its current position to the green platform",
  "Pick up the largest object and hold it over the red platform",
  "Find the smallest cube and place it on the blue platform",
  "Move the medium cube onto the platform that is not red or blue",
  "Place the large cube on the red platform, then release it",
  "Clear the blue platform by moving whatever is on it to the floor",
  "Move the small cube to the same platform where the large cube is sitting",
  "Stack the small cube on top of the medium cube.",
  "Stack the medium cube on top of the large cube.",
  "Build a three-cube tower with the largest at the bottom and smallest at the top.",
  "Place the large cube on top of the small cube (testing stability handling).",
  "Place the medium cube directly to the right of the red platform.",
  "Put the small cube on the green platform, next to the medium cube.",
  "Line up all three cubes in a straight row next to the blue platform.",
  "Place the small cube between the medium cube and the large cube.",
  "Move the large cube so it is touching the side of the red platform.",
  "Stack two cubes of any size onto the blue platform.",
  # "If the small cube is on the red platform, move it to the blue platform.",
  # "Move the medium cube to the red platform, then move the large cube to the green platform.",
  # "Stack the small cube on the medium cube, then move the e ntire stack to the blue platform.",
  # "Swap the positions of the small cube and the large cube.",
  # "Unstack the tower and place each cube on a different colored platform.",
  # "Put the small cube on the green platform only if the blue platform is empty.",
  # "Move all cubes off of the platforms and onto the floor.",
  # "Take the topmost cube off the stack and place it on the red platform.",
  # "Move the large cube to the blue platform without knocking over the small cube.",
  # "Place the medium cube on the platform that matches the color of the smallest cube's current platform.",
  # "Organize the environment so that every platform has exactly one cube on it.",
  # "Sort the cubes onto the platforms by size: smallest on red, medium on green, largest on blue.",
  # "Clear the red platform completely.",
  # "Reset the environment so that no cubes are touching each other or any platforms.",
  # "Hide the small cube by placing the large cube directly over or in front of it from the camera's view.",
  # "Create a stable two-cube stack on the green platform, leaving the large cube on the floor.",
  # "Move all objects as far away from the blue platform as possible.",
  # "Build the tallest tower possible on the red platform.",
  # "Reverse the order of the current cube stack.",
  # "Put all cubes on the same platform, ordered from smallest at the bottom to largest on top.",
]


def random_task():
    return TASKS[np.random.randint(len(TASKS))]


def pick_and_place():
  return f"Pick up the {random.choice(['small', 'medium', 'large'])} cube and place it on the {random.choice(['red', 'green', 'blue'])} platform"


def turn_to_cube():
  return f"Turn the robot to the {random.choice(['small', 'medium', 'large'])} cube"


def turn_right():
  return f"Turn the robot to the right"


tasks = {
  "pick_and_place": pick_and_place,
  "turn_to_cube": turn_to_cube,
  "turn_right": turn_right,
  "random": random_task
}