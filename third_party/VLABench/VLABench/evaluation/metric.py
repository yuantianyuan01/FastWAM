import numpy as np


def progress_score():
    """
    Compute the progress score of an episode. The progress score is defined as whether the neareast distance between the gripper to the target object is lower than a threshold.
    """
    pass

def soft_progress_score():
    """
    Compute the soft progress score of an episode. The progress score is defined as the (1 - distance/max_distance) between the gripper to the target object.
    """
    pass

def stage_score():
    """
    Compute the stage score of an episode. The stage score function is distinct for each task.
    """
    pass

def success_rate():
    pass