"""R2C — Official-success-derived sparse RL reward (project-side wrapper).

RoboTwin tasks expose the official task-completion predicate ``check_success()``.
Per MAINLINE-R2C §1, the formal RL reward is:

    r_t = 1.0  if official check_success() is True
          0.0  otherwise

This module reads only the official ``task.check_success()`` boolean. Privileged
state stays inside the environment's own predicate: there is no threshold
re-derivation, shaping, or progress variable.

Reward timing (§6): ``check_success()`` is already evaluated every inner physics
step inside ``Base_Task.take_action`` (``_base_task.py:1697``, right after each
``self.scene.step()``), and ``take_action`` early-returns once it flips
``eval_success``. The lifecycle audit verifies that the predicate is safe to call
at the wrapper boundary. We therefore use **per-step** reward
``r_t in {0, 1}`` (which collapses to terminal reward because the episode ends on
the first success).

Cross-task (§5): the same ``float(task.check_success())`` rule applies to every
RoboTwin task — no task-specific progress variable is hand-designed.
"""
from __future__ import annotations


def official_reward(task_env) -> float:
    """Official-success-derived sparse reward for one control step.

    Returns 1.0 iff the task's official completion predicate is currently True,
    else 0.0. Pure function of the task env's own success check.
    """
    return float(task_env.check_success())


def episode_return(success_rewards, gamma: float = 1.0) -> float:
    """Standard discounted return over a per-step sparse success reward sequence.

    With gamma=1.0 and a sparse {0,1} success reward, this is simply 1.0 if the
    episode ever succeeded, else 0.0. gamma is kept explicit (not swept).
    """
    R = 0.0
    for r in success_rewards:
        R = gamma * R + float(r)
    return R


# For a terminated episode with known success flag, the return reduces to the
# success indicator itself (gamma=1.0, sparse terminal reward).
def episode_return_from_success(success: bool, gamma: float = 1.0) -> float:
    """R_T = gamma**(T-t) * r_T  ->  success indicator when gamma=1.0."""
    return float(bool(success))
